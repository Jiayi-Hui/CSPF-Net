from __future__ import annotations

import importlib
import json
import sys


MODULES = ["datasets", "matplotlib", "nltk", "numpy", "pandas", "sklearn", "torch", "transformers"]


def main() -> None:
    status = {"python": sys.version, "modules": {}}
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
            status["modules"][module_name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            status["modules"][module_name] = f"missing: {exc}"
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
