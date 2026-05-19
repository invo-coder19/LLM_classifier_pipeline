"""Validate the golden dataset against its JSON schema."""
import json, sys, jsonschema
from pathlib import Path

schema = json.loads(Path("data/schema.json").read_text())
data   = json.loads(Path("data/golden_dataset.json").read_text())

try:
    jsonschema.validate(instance=data, schema=schema)
    print(f"✅ Dataset valid — {len(data['items'])} items")
except jsonschema.ValidationError as e:
    print(f"❌ Validation error: {e.message}")
    sys.exit(1)
