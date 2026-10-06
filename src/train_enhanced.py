"""Compatibility entry point for robot-separated diagnosis experiments."""
from pdm_runtime import make_feature_matrix as feat
from train_diagnosis import main


if __name__ == "__main__":
    raise SystemExit(main())
