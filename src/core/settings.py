import os
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

import yaml
from from_root import from_root

from src.core.exceptions import ConfigurationError
from src.core.logger import logging


DEFAULT_ENV = "dev"
ALLOWED_ENVS = {"dev", "uat", "prd"}
ENVIRONMENTS_DIR = os.path.join("configs", "environments")
SCHEMAS_DIR = os.path.join("configs", "schemas")


def _load_yaml(path: str) -> Dict[str, Any]:
    """Provide internal support for load yaml.

    Used by this module to keep the main workflow functions focused and readable.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            payload = yaml.safe_load(file)
        if not isinstance(payload, dict):
            raise ConfigurationError(f"YAML file must contain a mapping object: {path}")
        return payload
    except Exception as exc:
        raise ConfigurationError(f"Failed to load YAML: {path}") from exc


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Provide internal support for deep merge.

    Used by this module to keep the main workflow functions focused and readable.
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_nested(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Return nested derived from the provided inputs and runtime context.

    Encapsulates lookup and fallback logic in a single reusable call.
    """
    current: Any = config
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def require_nested(config: Dict[str, Any], path: str) -> Any:
    """Validate nested against expected constraints.

    Returns validation feedback or raises an error when required conditions are not met.
    """
    value = get_nested(config, path, default=None)
    if value is None:
        raise ConfigurationError(f"Missing required config key: {path}")
    return value


def resolve_environment(explicit_env: Optional[str] = None) -> str:
    """Return environment derived from the provided inputs and runtime context.

    Encapsulates lookup and fallback logic in a single reusable call.
    """
    env = str(explicit_env or os.getenv("APP_ENV", DEFAULT_ENV)).strip().lower()
    if env not in ALLOWED_ENVS:
        raise ConfigurationError(f"Unsupported APP_ENV '{env}'. Allowed values: {sorted(ALLOWED_ENVS)}")
    return env


@dataclass(frozen=True)
class RuntimeSettings:
    """Encapsulate `RuntimeSettings` behavior used by this module.

    Groups related state and methods into a reusable, testable abstraction.
    """
    env: str
    root_dir: str
    config: Dict[str, Any]

    @property
    def schema_path(self) -> str:
        """Execute schema path as part of the module workflow.

        Encapsulates a focused unit of pipeline logic for reuse and testing.
        Returns `str`.
        """
        schema_file = require_nested(self.config, "validation.schema_file")
        return os.path.join(self.root_dir, SCHEMAS_DIR, str(schema_file))


@lru_cache(maxsize=4)
def _load_settings_cached(env: str) -> RuntimeSettings:
    """Provide internal support for load settings cached.

    Used by this module to keep the main workflow functions focused and readable.
    """
    root_dir = from_root()
    base_path = os.path.join(root_dir, ENVIRONMENTS_DIR, "base.yaml")
    env_path = os.path.join(root_dir, ENVIRONMENTS_DIR, f"{env}.yaml")

    if not os.path.exists(base_path):
        raise ConfigurationError(f"Base environment config not found: {base_path}")
    if not os.path.exists(env_path):
        raise ConfigurationError(f"Environment config not found: {env_path}")

    base_config = _load_yaml(base_path)
    env_config = _load_yaml(env_path)
    merged_config = _deep_merge(base_config, env_config)

    merged_config["runtime"] = {
        "env": env,
        "root_dir": root_dir,
        "base_config_path": base_path,
        "env_config_path": env_path,
    }

    logging.info("Loaded runtime settings for env=%s", env)
    return RuntimeSettings(env=env, root_dir=root_dir, config=merged_config)


def load_settings(explicit_env: Optional[str] = None) -> RuntimeSettings:
    """Load settings from the configured source.

    Returns a parsed in-memory object for downstream processing.
    """
    env = resolve_environment(explicit_env)
    try:
        return _load_settings_cached(env)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise ConfigurationError("Unexpected error while loading runtime settings") from exc

