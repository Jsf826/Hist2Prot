import json
from pathlib import Path
from typing import Dict, Iterable


def load_config(path: str) -> Dict[str, object]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a JSON object: {config_path}")
    return payload


def flattened_defaults(config: Dict[str, object], *sections: Iterable[str]) -> Dict[str, object]:
    if len(sections) == 1 and not isinstance(sections[0], str):
        sections = tuple(sections[0])
    defaults: Dict[str, object] = {}
    for section in sections:
        value = config.get(section, {})
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"Config section {section!r} must be a JSON object.")
        defaults.update(value)
    return defaults
