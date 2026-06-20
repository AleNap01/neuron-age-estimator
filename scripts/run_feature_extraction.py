"""
run_feature_extraction.py — Entry point: estrazione feature v2 (155 feature).

Uso:
    python scripts/run_feature_extraction.py
"""

import numpy as np
from tqdm import tqdm

from brain_age.config import (
    FEATURES_TEST_V2,
    FEATURES_TRAIN_V2,
    LABELS_TRAIN_V2,
)
from brain_age.data import load_subject_volume, load_test_df, load_train_df
from brain_age.features import extract_features_v2


def main() -> None:
    train_df = load_train_df()
    test_df = load_test_df()

    print(f"Estrazione feature TRAIN ({len(train_df)} soggetti)...")
    X_train, y_train = [], []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
        volume = load_subject_volume(row["path"], split="train")
        X_train.append(extract_features_v2(volume))
        y_train.append(row["AGE"])

    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=np.float32)
    print(f"X_train shape: {X_train.shape}")

    print(f"\nEstrazione feature TEST ({len(test_df)} soggetti)...")
    X_test = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test"):
        volume = load_subject_volume(row["path"], split="test")
        X_test.append(extract_features_v2(volume))

    X_test = np.array(X_test)
    print(f"X_test shape: {X_test.shape}")

    np.save(FEATURES_TRAIN_V2, X_train)
    np.save(FEATURES_TEST_V2, X_test)
    np.save(LABELS_TRAIN_V2, y_train)

    print(f"\n[OK] Salvati in {FEATURES_TRAIN_V2.parent}")


if __name__ == "__main__":
    main()
