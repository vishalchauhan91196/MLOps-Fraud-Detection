import os
from typing import Any, Dict, List, Tuple

import yaml
from from_root import from_root

from src.core.exceptions import ConfigurationError
from src.core.settings import (
    ALLOWED_ENVS,
    ENVIRONMENTS_DIR,
    _deep_merge,
    _load_yaml,
    get_nested,
)
from src.core.logger import logging


TYPE_MAP = {
    "str": str,
    "int": int,
    "float": (float, int),
    "bool": bool,
    "list": list,
    "dict": dict,
}


def _load_contract(contract_path: str) -> List[Dict[str, Any]]:
    """Load and validate the runtime contract file.

    The contract must be a YAML mapping with a `required_keys` list. Each list
    item is expected to describe a required configuration key with attributes
    such as `path` and `type`.
    """
    with open(contract_path, "r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict) or "required_keys" not in payload:
        raise ConfigurationError(f"Invalid contract file: {contract_path}")
    required = payload["required_keys"]
    if not isinstance(required, list):
        raise ConfigurationError(f"Invalid contract required_keys type: {contract_path}")
    return required


def _validate_required_key(config: Dict[str, Any], path: str, type_name: str, env: str) -> List[str]:
    """Validate one required config key against the contract.

    Checks that the key exists, matches the expected type declared in the
    contract, and is not an empty string/list/dict when present.
    Returns environment-scoped validation error messages.
    """
    errors: List[str] = []
    value = get_nested(config, path, default=None)
    if value is None:
        errors.append(f"[{env}] missing required key: {path}")
        return errors

    expected = TYPE_MAP.get(type_name)
    if expected is None:
        errors.append(f"[{env}] unsupported contract type '{type_name}' for key {path}")
        return errors
    if not isinstance(value, expected):
        errors.append(f"[{env}] invalid type for {path}: expected {type_name}, got {type(value).__name__}")
        return errors

    if isinstance(value, str) and not value.strip():
        errors.append(f"[{env}] empty string value for key: {path}")
    if isinstance(value, list) and len(value) == 0:
        errors.append(f"[{env}] empty list value for key: {path}")
    if isinstance(value, dict) and len(value) == 0:
        errors.append(f"[{env}] empty dict value for key: {path}")
    return errors


def _validate_semantics(config: Dict[str, Any], env: str) -> List[str]:
    """Run semantic validations that go beyond type checks.

    Current rules verify ingestion split bounds and the type of the pipeline
    random state. Returns all semantic errors found for the environment.
    """
    errors: List[str] = []
    test_size = float(get_nested(config, "ingestion.test_size", default=-1))
    if not 0.0 < test_size < 1.0:
        errors.append(f"[{env}] ingestion.test_size must be between 0 and 1, got {test_size}")
    random_state = get_nested(config, "pipeline.random_state", default=None)
    if not isinstance(random_state, int):
        errors.append(f"[{env}] pipeline.random_state must be int")
    return errors


def lint_all_environments() -> Tuple[bool, List[str]]:
    """Lint all configured environments using contract and semantic checks.

    Loads `base.yaml`, merges each environment override (`dev`, `uat`, `prd`),
    validates required contract keys, and then applies semantic rules.
    Returns `(is_valid, errors)` where `errors` contains all violations.
    """
    root_dir = from_root()
    base_path = os.path.join(root_dir, ENVIRONMENTS_DIR, "base.yaml")
    contract_path = os.path.join(root_dir, "configs", "contracts", "runtime_required.yaml")

    base_config = _load_yaml(base_path)
    contract_items = _load_contract(contract_path)

    errors: List[str] = []
    for env in sorted(ALLOWED_ENVS):
        env_path = os.path.join(root_dir, ENVIRONMENTS_DIR, f"{env}.yaml")
        env_config = _load_yaml(env_path)
        merged = _deep_merge(base_config, env_config)

        for item in contract_items:
            if not isinstance(item, dict):
                errors.append(f"[{env}] invalid contract row; expected mapping")
                continue
            path = item.get("path")
            type_name = item.get("type")
            if not isinstance(path, str) or not isinstance(type_name, str):
                errors.append(f"[{env}] contract row missing path/type")
                continue
            errors.extend(_validate_required_key(merged, path, type_name, env))

        errors.extend(_validate_semantics(merged, env))

    return len(errors) == 0, errors


def main() -> None:
    """Execute config linting as a standalone pipeline stage.

    Logs each validation error, raises `ConfigurationError` on failure, and
    emits a success log when all environments pass lint checks.
    """
    try:
        ok, errors = lint_all_environments()
        if not ok:
            for err in errors:
                logging.error(err)
            raise ConfigurationError("Configuration lint failed")
        logging.info("Configuration lint passed for all environments: %s", sorted(ALLOWED_ENVS))
    except Exception:
        logging.exception("Configuration lint stage failed")
        raise


if __name__ == "__main__":
    main()

