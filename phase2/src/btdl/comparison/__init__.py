"""Cross-model comparison pipeline (D14, D15, D18, D19).

Two modes:
  - "test": the real, post-unblinding comparison, built ONLY from exported
    phase2/artifacts/results/<model>/seed<k>/ directories plus the Phase 1
    baseline's test predictions. Gated on unblinding_status being READY.
  - "val-preview": an exploratory preview built from registered models'
    LOCAL eval/val/ directories, every output clearly labelled
    "VALIDATION PREVIEW -- NOT RESULTS", written to the gitignored
    phase2/runs/_preview/comparison/. Never reads or constructs the test
    split.
"""
