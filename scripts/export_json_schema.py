"""
Dump Pydantic schemas to JSON Schema for the TS side (W2).

Usage:
    python3 -m pipeline.scripts.export_json_schema

Writes:
    web/src/lib/matching/schemas/caption_intelligence.schema.json
    web/src/lib/matching/schemas/transcript_intelligence.schema.json
    web/src/lib/matching/schemas/audience_intelligence.schema.json

Run this whenever the Pydantic schemas change so the TS Zod mirrors
regenerate (via `make schemas` or manually).
"""

from __future__ import annotations

import json
import pathlib
import sys

from pipeline.schemas.intelligence import (
    CaptionIntelligencePayload,
    TranscriptIntelligencePayload,
    AudienceIntelligencePayload,
)


DEFAULT_OUTPUT_DIR = pathlib.Path(
    "web/src/lib/matching/schemas"
).resolve()


SCHEMA_MAP = {
    "caption_intelligence": CaptionIntelligencePayload,
    "transcript_intelligence": TranscriptIntelligencePayload,
    "audience_intelligence": AudienceIntelligencePayload,
}


def main(output_dir: pathlib.Path = DEFAULT_OUTPUT_DIR) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in SCHEMA_MAP.items():
        schema = model.model_json_schema()
        schema["$id"] = f"https://influencer.local/schemas/{name}.schema.json"
        schema["title"] = model.__name__
        target = output_dir / f"{name}.schema.json"
        target.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    dest = (
        pathlib.Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else DEFAULT_OUTPUT_DIR
    )
    sys.exit(main(dest))
