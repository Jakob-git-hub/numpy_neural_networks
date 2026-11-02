"""Evaluate ten selected hyperparameter configurations on Fashion-MNIST.

Example:
    python hypersearch.py --epochs 8 --max-samples 20000
"""

from __future__ import annotations

import argparse
import json
import ssl
from pathlib import Path

import numpy as np

from neural_class import dataset, neural_net


CONFIGURATIONS = (
    {"architecture": [[128, "relu"], [10, "softmax"]], "learning_rate": 0.001, "batch_size": 64},
    {"architecture": [[256, "relu"], [10, "softmax"]], "learning_rate": 0.001, "batch_size": 64},
    {"architecture": [[256, "relu"], [10, "softmax"]], "learning_rate": 0.003, "batch_size": 64},
    {"architecture": [[512, "relu"], [10, "softmax"]], "learning_rate": 0.001, "batch_size": 64},
    {"architecture": [[512, "relu"], [10, "softmax"]], "learning_rate": 0.003, "batch_size": 64},
    {"architecture": [[256, "relu"], [128, "relu"], [10, "softmax"]], "learning_rate": 0.001, "batch_size": 64},
    {"architecture": [[256, "relu"], [128, "relu"], [10, "softmax"]], "learning_rate": 0.003, "batch_size": 64},
    {"architecture": [[512, "relu"], [256, "relu"], [10, "softmax"]], "learning_rate": 0.001, "batch_size": 128},
    {"architecture": [[256, "arctan_sin"], [128, "arctan_sin"], [10, "softmax"]], "learning_rate": 0.001, "batch_size": 64},
    {"architecture": [[256, "relu_sin"], [128, "relu_sin"], [10, "softmax"]], "learning_rate": 0.001, "batch_size": 64},
)


def load_fashion_mnist(max_samples: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load and optionally sample the same combined data as neuralclass_MNIST."""
    try:
        from tensorflow.keras.datasets import fashion_mnist  # type: ignore
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "TensorFlow is required to load Fashion-MNIST. "
            "Install it with: python -m pip install tensorflow"
        ) from error

    ssl._create_default_https_context = ssl._create_unverified_context
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_all = np.concatenate((x_train, x_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)

    if max_samples is not None and max_samples < len(x_all):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(x_all), size=max_samples, replace=False)
        x_all = x_all[indices]
        y_all = y_all[indices]

    return x_all, y_all


def run_trial(
    configuration: dict,
    x_all: np.ndarray,
    y_all: np.ndarray,
    epochs: int,
    split: float,
    seed: int,
) -> dict:
    """Train and evaluate one configuration on a fresh model and dataset."""
    np.random.seed(seed)

    data = dataset(x_all.copy(), y_all.copy())
    data.prepare(output_dim=10, split=split, batch_size=configuration["batch_size"])

    model = neural_net(data.x.shape[1], configuration["architecture"])
    model.fit(
        data,
        learning_rate=configuration["learning_rate"],
        epochs=epochs,
    )
    loss, accuracy = model.evaluate(data)

    return {
        **configuration,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "seed": seed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("hypersearch_results.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if not 0 < args.split < 1:
        raise ValueError("--split must be between 0 and 1")

    x_all, y_all = load_fashion_mnist(args.max_samples, args.seed)
    results = []

    for trial, configuration in enumerate(CONFIGURATIONS):
        print(f"\nTrial {trial + 1}/{len(CONFIGURATIONS)}: {configuration}")
        result = run_trial(
            configuration,
            x_all,
            y_all,
            epochs=args.epochs,
            split=args.split,
            seed=args.seed + trial,
        )
        results.append(result)
        print(
            f"validation loss={result['loss']:.4f}, "
            f"validation accuracy={result['accuracy']:.2%}"
        )

    results.sort(key=lambda result: (-result["accuracy"], result["loss"]))
    args.output.write_text(json.dumps(results, indent=2) + "\n")

    print("\nBest configuration:")
    print(json.dumps(results[0], indent=2))
    print(f"\nAll results written to {args.output}")


if __name__ == "__main__":
    main()