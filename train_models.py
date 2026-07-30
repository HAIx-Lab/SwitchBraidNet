"""
Train models with Quantization-Aware Training (QAT).

Entry point for training all configured models across BCI paradigms and
precision levels. Iterates over the Cartesian product of
(models × modes × precisions) defined in ``config.py``.

Usage:
    python train_models.py
"""

import time
from config import CONFIG
from utils.registry import get_model_spec
from utils.engine import train_model


def main():
    print("Starting Training Workflow...")
    print(f"Models: {CONFIG['models']}")
    print(f"Modes: {CONFIG['modes']}")
    print(f"Precisions: {CONFIG['precisions']}")
    print("-" * 40)

    for model_name in CONFIG["models"]:
        try:
            spec = get_model_spec(model_name)
            for mode in CONFIG["modes"]:
                for bits in CONFIG["precisions"]:
                    train_model(spec, mode, bits, CONFIG)

                    if CONFIG["cooldown_seconds"] > 0:
                        print(f"Cooling down for {CONFIG['cooldown_seconds']}s...")
                        time.sleep(CONFIG["cooldown_seconds"])
        except Exception as e:
            print(f"Error training {model_name}: {e}")


if __name__ == "__main__":
    main()
