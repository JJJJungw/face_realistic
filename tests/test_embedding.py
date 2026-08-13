import numpy as np

from face_realistic.identity.embedding import cosine_similarity, l2_normalize


def test_l2_normalize() -> None:
    result = l2_normalize(np.array([3.0, 4.0]))
    np.testing.assert_allclose(result, [0.6, 0.8])


def test_cosine_similarity() -> None:
    assert np.isclose(cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])), 1.0)
    assert np.isclose(cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0)


def test_l2_normalize_rejects_zero_vector() -> None:
    try:
        l2_normalize(np.zeros(3))
    except ValueError:
        pass
    else:
        raise AssertionError("zero vector must fail normalization")
