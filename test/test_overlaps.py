import pytest
from trees.data import Data


def test_intersection():
    D = Data()

    assert D.intersection_range((1, 5), (3, 7)) == (3, 5)

    assert D.intersection((1, 5), (3, 7)) == 2
    assert D.intersection((1, 2), (3, 7)) == 0

    with pytest.raises(ValueError):
        D.intersection_range((1, 2), (3, 7))

    assert D.is_subset((1, 5), (3, 4)) == True
    assert D.is_subset((-3, 5), (3, 5)) == True
    assert D.is_subset((1, 5), (3, 7)) == False
    assert D.is_subset((1, 5), (10, 12)) == False
    assert D.is_subset((1, 5), (-10, 15)) == False


def test_overlaps():
    D = Data()
    assert D.jaccard_between_intervals((0, 10), (5, 10)) == 0.5

    cond1 = {"f1": (0, 10), "f2": (5, 10)}
    cond2 = {"f1": (0, 10), "f2": (5, 10)}

    assert D.jaccard(cond1, cond2) == 1.0

    cond1 = {"f1": (0, 10), "f2": (0, 10)}
    cond2 = {"f1": (0, 10), "f2": (5, 10)}

    assert D.jaccard(cond1, cond2) == 0.75

    cond3 = {"f3": (0, 10)}
    assert D.jaccard(cond1, cond3) == 0
