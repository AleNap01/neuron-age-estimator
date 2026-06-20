"""
run_train_classical.py — Entry point: training e valutazione modelli ML classici.

Uso:
    python scripts/run_train_classical.py
"""

import numpy as np

from brain_age.config import (
    FEATURES_TEST_V2,
    FEATURES_TRAIN_V2,
    LABELS_TRAIN_V2,
    MODEL_V2_PATH,
    SUBMISSION_V2_PATH,
)
from brain_age.data import load_test_df
from brain_age.training.train_classical import (
    evaluate_models,
    generate_submission,
    select_best_model,
    train_final_model,
)


def main() -> None:
    X_train = np.load(FEATURES_TRAIN_V2)
    X_test = np.load(FEATURES_TEST_V2)
    y_train = np.load(LABELS_TRAIN_V2)

    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    print("\nValutazione modelli (5-fold cross-validation)...")
    results = evaluate_models(X_train, y_train)

    best_name, best_info = select_best_model(results)
    print(f"\nMiglior modello: {best_name} (MAE CV = {best_info['mae_mean']:.4f})")

    print(f"\nTraining finale di {best_name} sull'intero training set...")
    train_mae = train_final_model(best_info["pipeline"], X_train, y_train, MODEL_V2_PATH)
    print(f"MAE sul training set: {train_mae:.4f}")
    print(f"Modello salvato in: {MODEL_V2_PATH}")

    test_df = load_test_df()
    submission = generate_submission(best_info["pipeline"], X_test, test_df, SUBMISSION_V2_PATH)
    print(f"\nSubmission salvata in: {SUBMISSION_V2_PATH}")
    print(submission.head())


if __name__ == "__main__":
    main()
