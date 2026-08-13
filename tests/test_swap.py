from face_realistic.swap.baseline import _largest_face


class FakeFace:
    def __init__(self, bbox: list[float]) -> None:
        self.bbox = bbox


def test_largest_face() -> None:
    small = FakeFace([0, 0, 10, 10])
    large = FakeFace([0, 0, 20, 30])
    assert _largest_face([small, large]) is large


def test_largest_face_rejects_empty_list() -> None:
    try:
        _largest_face([])
    except RuntimeError:
        pass
    else:
        raise AssertionError("empty face list must fail")
