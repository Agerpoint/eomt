# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

"""Metadata-only MLflow model wrapper for EoMT weights."""

import mlflow


class EoMTC(mlflow.pyfunc.PythonModel):
    """Package EoMT weights for model registration, not model serving."""

    def __init__(self, pt_file: str) -> None:
        self.pt_file = pt_file

    def load_context(self, context) -> None:
        pass
