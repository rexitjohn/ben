from ben.data import FEATURE_COLUMNS, TARGET_COLUMN, make_synthetic_canine_dataset


def test_synthetic_dataset_shape_and_columns():
    df = make_synthetic_canine_dataset(rows=200, random_state=7)
    assert len(df) == 200
    assert TARGET_COLUMN in df.columns
    for c in FEATURE_COLUMNS: assert c in df.columns

def test_target_is_binary():
    df = make_synthetic_canine_dataset(rows=200, random_state=8)
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})
