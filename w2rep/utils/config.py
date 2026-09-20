from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A recursively wrapped dictionary with attribute access."""

    def __init__(self, values: dict[str, Any] | None = None) -> None:
        super().__init__()
        for key, value in (values or {}).items():
            self[key] = self._wrap(value)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)

    def plain(self) -> dict[str, Any]:
        def unwrap(value: Any) -> Any:
            if isinstance(value, Config):
                return {key: unwrap(item) for key, item in value.items()}
            if isinstance(value, list):
                return [unwrap(item) for item in value]
            return value

        return unwrap(self)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_values(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path in stack:
        raise ValueError(f"Recursive configuration inheritance: {stack + (path,)}")
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    base_reference = values.pop("base", None)
    if base_reference is None:
        return values
    base_path = Path(base_reference).expanduser()
    if not base_path.is_absolute():
        base_path = (path.parent / base_path).resolve()
    return _deep_merge(_load_values(base_path, stack + (path,)), values)


def load_config(path: str | Path) -> tuple[Config, str]:
    path = Path(path).expanduser().resolve()
    values = _load_values(path)
    config = Config(values)
    _validate(config)
    canonical = json.dumps(config.plain(), sort_keys=True, separators=(",", ":"))
    return config, hashlib.sha256(canonical.encode()).hexdigest()


def _validate(config: Config) -> None:
    required = ("output_dir", "data", "model", "mask", "objective", "ema", "train")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing configuration keys: {missing}")
    if config.data.frames < 2:
        raise ValueError("W2Rep pretraining needs at least two frames")
    if config.data.cross_frame_targets < 1 and config.objective.cross_frame:
        raise ValueError("cross_frame_targets must be positive when cross-frame loss is enabled")
    if not (config.objective.same_frame or config.objective.cross_frame):
        raise ValueError("At least one prediction objective must be enabled")
    if config.objective.z_l2 < 0:
        raise ValueError("z_l2 must be non-negative")
    z_context = config.objective.get("z_context", "all")
    if z_context not in {"all", "exclude_targets", "exclude_random"}:
        raise ValueError(f"Unknown objective.z_context={z_context!r}")
    if config.data.image_size % config.model.patch_size:
        raise ValueError("image_size must be divisible by patch_size")
    if str(config.train.amp).lower() not in {"bf16", "fp32"}:
        raise ValueError("train.amp must be 'bf16' or 'fp32'")
    if config.train.total_steps <= 0 or config.train.batch_size_per_gpu <= 0:
        raise ValueError("Training steps and batch size must be positive")
