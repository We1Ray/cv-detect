"""Traditional machine learning classifiers for industrial inspection.

Provides SVM, KNN, MLP, and Random Forest classifiers with a unified
interface for training, inference, and model persistence. Designed for
small-sample scenarios common in industrial inspection.

Key components
--------------
- ``ClassifierModel`` -- serialisable model container.
- ``FeatureVector`` -- standardised feature input.
- ``ClassifierTrainer`` -- training and cross-validation.
- ``ClassifierInference`` -- prediction and confidence scoring.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Single classification result."""
    class_id: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClassifierModel:
    """Serialisable classifier model container."""
    model_type: str  # "svm", "knn", "mlp", "random_forest"
    class_names: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    model_params: Dict[str, Any] = field(default_factory=dict)
    _sklearn_model: Any = field(default=None, repr=False)
    _scaler: Any = field(default=None, repr=False)

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "model_type": self.model_type,
            "class_names": self.class_names,
            "feature_names": self.feature_names,
            "model_params": self.model_params,
            "sklearn_model": self._sklearn_model,
            "scaler": self._scaler,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Classifier saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path], *, trusted: bool = False) -> "ClassifierModel":
        """Load model from disk.

        WARNING: Uses pickle deserialization. Only load files you trust.

        Parameters
        ----------
        trusted : bool
            Must be True to acknowledge the security risk of pickle loading.
        """
        if not trusted:
            raise ValueError(
                "Loading pickle models is a security risk. "
                "Pass trusted=True to acknowledge this risk."
            )
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        model = cls(
            model_type=data["model_type"],
            class_names=data["class_names"],
            feature_names=data.get("feature_names", []),
            model_params=data.get("model_params", {}),
        )
        model._sklearn_model = data["sklearn_model"]
        model._scaler = data.get("scaler")
        logger.info(
            "Classifier loaded from %s (type=%s)", path, model.model_type,
        )
        return model


class ClassifierTrainer:
    """Train traditional ML classifiers.

    Parameters
    ----------
    model_type : str
        "svm", "knn", "mlp", or "random_forest".
    class_names : list of str
        Ordered list of class labels.
    normalize : bool
        Whether to apply StandardScaler to features.
    """

    def __init__(
        self,
        model_type: str = "svm",
        class_names: Optional[List[str]] = None,
        normalize: bool = True,
        **model_kwargs: Any,
    ) -> None:
        self._model_type = model_type
        self._class_names = class_names or []
        self._normalize = normalize
        self._model_kwargs = model_kwargs

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> ClassifierModel:
        """Train classifier on feature matrix and labels.

        Parameters
        ----------
        features : ndarray (N, D)
            Feature matrix.
        labels : ndarray (N,)
            Integer class labels.
        feature_names : list of str, optional
            Names for each feature dimension.
        """
        from sklearn.preprocessing import StandardScaler

        if len(labels) == 0:
            raise ValueError("labels array must not be empty")

        if not self._class_names:
            self._class_names = [str(i) for i in range(int(labels.max()) + 1)]

        scaler = None
        X = features.astype(np.float32)
        if self._normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        clf = self._create_sklearn_model()
        t0 = time.perf_counter()
        clf.fit(X, labels)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Trained %s classifier on %d samples in %.1f ms",
            self._model_type, len(labels), elapsed,
        )

        model = ClassifierModel(
            model_type=self._model_type,
            class_names=self._class_names,
            feature_names=feature_names or [],
            model_params=self._model_kwargs,
        )
        model._sklearn_model = clf
        model._scaler = scaler
        return model

    def cross_validate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        cv: int = 5,
    ) -> Dict[str, float]:
        """Run k-fold cross-validation and return metrics."""
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        X = features.astype(np.float32)
        if self._normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        clf = self._create_sklearn_model()
        scores = cross_val_score(clf, X, labels, cv=cv, scoring="accuracy")
        return {
            "accuracy_mean": float(np.mean(scores)),
            "accuracy_std": float(np.std(scores)),
            "fold_scores": scores.tolist(),
        }

    def _create_sklearn_model(self) -> Any:
        if self._model_type == "svm":
            from sklearn.svm import SVC
            return SVC(probability=True, **self._model_kwargs)
        elif self._model_type == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            k = self._model_kwargs.get("n_neighbors", 5)
            # Remove from kwargs before passing to constructor to avoid duplicate
            kw = {k_: v for k_, v in self._model_kwargs.items() if k_ != "n_neighbors"}
            return KNeighborsClassifier(
                n_neighbors=k, **kw,
            )
        elif self._model_type == "mlp":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(max_iter=500, **self._model_kwargs)
        elif self._model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100, **self._model_kwargs,
            )
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")


class ClassifierInference:
    """Run inference with a trained classifier.

    Parameters
    ----------
    model : ClassifierModel
        Trained model container.
    """

    def __init__(self, model: ClassifierModel) -> None:
        self._model = model
        if model._sklearn_model is None:
            raise ValueError("Model has no trained sklearn model.")

    def predict(self, features: np.ndarray) -> ClassificationResult:
        """Classify a single feature vector.

        Parameters
        ----------
        features : ndarray (D,) or (1, D)
        """
        X = features.reshape(1, -1).astype(np.float32)
        if self._model._scaler is not None:
            X = self._model._scaler.transform(X)

        clf = self._model._sklearn_model
        class_id = int(clf.predict(X)[0])
        class_name = (
            self._model.class_names[class_id]
            if class_id < len(self._model.class_names)
            else str(class_id)
        )

        probs: Dict[str, float] = {}
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(X)[0]
            for i, name in enumerate(self._model.class_names):
                if i < len(p):
                    probs[name] = float(p[i])
            confidence = float(p[class_id]) if class_id < len(p) else 0.0
        elif hasattr(clf, "decision_function"):
            df = clf.decision_function(X)[0]
            confidence = float(
                1.0
                / (
                    1.0
                    + np.exp(
                        -abs(
                            float(df) if np.isscalar(df) else float(np.max(df))
                        )
                    )
                )
            )
        else:
            confidence = 1.0

        return ClassificationResult(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            probabilities=probs,
        )

    def predict_batch(
        self, features: np.ndarray,
    ) -> List[ClassificationResult]:
        """Classify a batch of feature vectors (N, D)."""
        return [self.predict(features[i]) for i in range(len(features))]

    def feature_importance(self) -> Optional[Dict[str, float]]:
        """Return feature importance if available (e.g., Random Forest)."""
        clf = self._model._sklearn_model
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            names = self._model.feature_names or [
                f"f{i}" for i in range(len(imp))
            ]
            return {names[i]: float(imp[i]) for i in range(len(imp))}
        return None
