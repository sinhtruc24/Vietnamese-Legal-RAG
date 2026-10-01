"""QLoRA fine-tuning of a small instruct LLM for grounded, cited legal answers.

Single GPU (Colab / Kaggle T4):
    python training/train_qlora.py --model Qwen/Qwen2.5-3B-Instruct --output-dir outputs/qwen-legal-lora

Two GPUs (Kaggle "T4 x2"), one model replica per GPU with DDP, about 2x faster:
    torchrun --nproc_per_node 2 training/train_qlora.py ...

A checkpoint is written every --save-steps; rerun with --resume to continue an
interrupted run (e.g. in a new Kaggle session, after copying outputs/ back).

The data is in TRL's conversational prompt/completion format, so the loss is
computed on the assistant answer only, not on the (long) legal context.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
if WORLD_SIZE == 1:
    # Without torchrun, hide extra GPUs: with 2 visible GPUs Trainer wraps the model in
    # nn.DataParallel, and replicating a bitsandbytes 4-bit model crashes ("illegal memory access").
    # Must be set before torch initialises CUDA.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("data/sft/train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/sft/val.jsonl"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen-legal-lora"))
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2, help="per GPU")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="for a single GPU; divided by the number of GPUs so the effective batch stays the same")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in --output-dir")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    # T4 has no bf16 support; fall back to fp16 there.
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    grad_accum = max(1, args.grad_accum // WORLD_SIZE)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        ),
        dtype=compute_dtype,
        # Whole model on this process's GPU: a 4-bit 3B model fits on one T4. Splitting it across
        # GPUs ("auto") is slower and wraps forward() in accelerate hooks that break TRL.
        device_map={"": LOCAL_RANK} if torch.cuda.is_available() else None,
    )

    data = load_dataset("json", data_files={"train": str(args.train), "validation": str(args.val)})
    data = data.select_columns(["prompt", "completion"])

    steps_per_epoch = -(-len(data["train"]) // (args.batch_size * grad_accum * WORLD_SIZE))
    extra = {}
    if "loss_type" in SFTConfig.__dataclass_fields__:
        # Newer TRL defaults to "chunked_nll", which patches model.forward and fails on hooked/quantized
        # models; plain NLL is the same objective.
        extra["loss_type"] = "nll"

    config = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * steps_per_epoch * args.epochs)),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        max_length=args.max_length,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        # Keep every checkpoint (~120 MB each) and pick the best one ourselves below:
        # load_best_model_at_end reloads the adapter through peft's tensor-parallel path, which
        # crashes on some peft/transformers combinations (ImportError: EmbeddingParallel).
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
        **extra,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    if trainer.is_world_process_zero():
        trainer.model.print_trainable_parameters()
        print(f"GPUs: {WORLD_SIZE} | per-GPU batch {args.batch_size} x accum {grad_accum} | "
              f"{steps_per_epoch} steps/epoch")

    resume = args.resume and any(args.output_dir.glob("checkpoint-*"))
    if args.resume and not resume and trainer.is_world_process_zero():
        print(f"--resume: no checkpoint in {args.output_dir}, starting from scratch")
    trainer.train(resume_from_checkpoint=True if resume else None)

    adapter_dir = args.output_dir / "adapter"
    best = trainer.state.best_model_checkpoint
    if best is None:  # no evaluation ran: keep the final weights
        trainer.save_model(str(adapter_dir))
    if trainer.is_world_process_zero():
        if best is not None:
            export_adapter(Path(best), adapter_dir)
            print(f"best checkpoint: {best} (eval_loss={trainer.state.best_metric:.4f})")
        tokenizer.save_pretrained(str(adapter_dir))
        history = [h for h in trainer.state.log_history if "eval_loss" in h]
        (args.output_dir / "eval_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"adapter saved to {adapter_dir}")


def export_adapter(checkpoint: Path, adapter_dir: Path) -> None:
    """Copy the LoRA weights + config of a Trainer checkpoint into a standalone adapter folder."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    files = list(checkpoint.glob("adapter_*"))
    if not files:
        raise FileNotFoundError(f"no adapter_* files in {checkpoint}")
    for file in files:
        shutil.copy(file, adapter_dir / file.name)


if __name__ == "__main__":
    main()
