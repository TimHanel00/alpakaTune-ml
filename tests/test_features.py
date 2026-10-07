from alpakatune_ml.features import (
    DIMENSION_FEATURE_NAMES,
    dimension_features,
    fnv1a64,
    signed_hash_features,
)


def test_dimension_schema_matches_native_contract():
    values = dimension_features(
        name="blockSize",
        value=128,
        domain=[64, 128, 256],
        dimension_position=1,
        dimension_count=3,
        kind="launch",
    )
    assert len(values) == len(DIMENSION_FEATURE_NAMES) == 18
    assert values[0] == 0.5
    assert values[8] == 1.0
    assert sum(abs(value) for value in values[10:]) == 1.0


def test_fnv_hashing_is_stable_and_signed_one_hot():
    assert fnv1a64("blockSize") == 13386963988543079947
    first = signed_hash_features("blockSize")
    second = signed_hash_features("blockSize")
    assert first == second
    assert sum(value != 0 for value in first) == 1
    assert next(value for value in first if value) in {-1.0, 1.0}
