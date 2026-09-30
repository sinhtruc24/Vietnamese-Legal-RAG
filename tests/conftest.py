import pytest

from legalrag.data import Article, chunk_corpus


@pytest.fixture
def articles() -> dict[str, Article]:
    rows = [
        Article("100/2019/nđ-cp+5", "Điều 5. Xử phạt người điều khiển xe mô tô",
                "Phạt tiền từ 100.000 đồng đến 200.000 đồng đối với người điều khiển xe mô tô không mang theo giấy phép lái xe."),
        Article("100/2019/nđ-cp+6", "Điều 6. Không đội mũ bảo hiểm",
                "Phạt tiền đối với người điều khiển, người ngồi trên xe mô tô không đội mũ bảo hiểm khi tham gia giao thông."),
        Article("45/2019/qh14+35", "Điều 35. Quyền đơn phương chấm dứt hợp đồng lao động của người lao động",
                "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước cho người sử dụng lao động ít nhất 45 ngày."),
        Article("01/2009/tt-bnn+1", "Điều 1. Phạm vi áp dụng",
                "Thông tư này hướng dẫn tuần tra, canh gác bảo vệ đê điều trong mùa lũ."),
    ]
    return {a.id: a for a in rows}


@pytest.fixture
def chunks(articles):
    return chunk_corpus(articles.values())
