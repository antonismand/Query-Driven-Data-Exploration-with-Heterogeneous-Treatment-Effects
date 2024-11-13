from trees.data import Data
from trees.parser import Predicate

D = Data()
D.generate(seed=42)


def test_P():
    p1 = Predicate(p="feature_0>5 AND feature_1>10", D=D)

    assert p1.conditions == [
        [{"feature_0": (5.0, 3.728)}],
        [{"feature_1": (10.0, 3.942)}],
    ]

    p2 = Predicate(
        p="(feature_1 > 1 OR feature_2 < 3) AND feature_4 > 3 AND feature_5 > 3 AND feature_1 between 1 and 2 AND (feature_0 < 0 OR feature_0 > 3 OR feature_3 < 3)",
        D=D,
    )

    assert p2.conditions == [
        [{"feature_1": (1.0, 3.942)}, {"feature_2": (-3.94, 3)}],
        [{"feature_4": (3.0, 3.594)}],
        [{"feature_5": (3.0, 3.926)}],
        [{"feature_1": (1.0, 2.0)}],
        [
            {"feature_0": (-4.295, 0.0)},
            {"feature_0": (3.0, 3.728)},
            {"feature_3": (-3.632, 3.0)},
        ],
    ]

    assert (
        p2.includes(
            {"feature_1": (1.5, 2), "feature_0": (3.1, 3.5)},
        )
        == True
    )
    assert p2.includes({"feature_6": (0, 1)}) == False
    # assert p2.includes({"feature_5": (3, 4)}) == False
    assert (
        p2.includes(
            {
                "feature_1": (1, 2),
                "feature_0": (3.1, 3.5),
                "feature_2": (0, 1),
                "feature_3": (0, 2),
            }
        )
        == True
    )

    assert p2.includes({"feature_0": (3.0, 3.5)}) == False

    p3 = Predicate(p="feature_2 > 0", D=D)
    assert p3.includes({"feature_1": (0, 1)}) == True


def test_P_with_HTE():
    p1 = Predicate(p="feature_0 > 0", D=D)
    assert p1.includes({"feature_1": (0, 1)}) == False

    assert p1.includes({"feature_0": (0, 1)}) == True
    assert p1.includes({"feature_0": (0, 3), "feature_2": (0, 1)})

    p2 = Predicate(p="feature_0 > 0 AND feature_2 > 0", D=D)
    assert p2.includes({"feature_0": (0, 1)}) == True

    p3 = Predicate(p="feature_0 > 0 AND feature_1 > 0 AND feature_2>0", D=D)
    assert p3.includes({"feature_0": (0, 1)}) == False
    assert p3.includes({"feature_0": (0, 1), "feature_1": (0, 1)}) == True


def test_P_without_HTE():
    p1 = Predicate(p="feature_3 > 0", D=D)

    assert p1.includes({"feature_1": (0, 1)}) == True
    assert p1.includes({"feature_3": (0, 1)}) == True
