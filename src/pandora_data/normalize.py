from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from .schema import finalize_case


def _message(role: str, content: Any, annotations: dict[str, Any] | None = None) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    result: dict[str, Any] = {"role": role, "content": text}
    clean_annotations = {
        key: value for key, value in (annotations or {}).items()
        if value not in (None, "", "n/a")
    }
    if clean_annotations:
        result["annotations"] = clean_annotations
    return result


def normalize_psydial(path: Path, revision: str = "unknown") -> Iterator[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    role_map = {"user": "client", "assistant": "counselor"}
    for record_index, record in enumerate(records):
        systems = [m.get("content", "").strip() for m in record["messages"] if m.get("role") == "system"]
        messages = [
            normalized
            for raw in record["messages"]
            if raw.get("role") in role_map
            if (normalized := _message(role_map[raw["role"]], raw.get("content"))) is not None
        ]
        yield finalize_case({
            "case_id": f"psydial_d4:{record_index:04d}",
            "source": "psydial_d4",
            "language": "zh",
            "messages": messages,
            "profile": None,
            "metadata": {"system_prompt": "\n\n".join(systems)},
            "provenance": {
                "source_file": str(path.as_posix()),
                "source_record_id": str(record_index),
                "revision": revision,
            },
        })


def normalize_annomi(path: Path, revision: str = "unknown") -> Iterator[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["transcript_id"]].append(row)
    for transcript_id in sorted(grouped, key=lambda value: int(value)):
        rows = sorted(grouped[transcript_id], key=lambda row: int(row["utterance_id"]))
        first = rows[0]
        messages = []
        for row in rows:
            role = "counselor" if row["interlocutor"] == "therapist" else "client"
            normalized = _message(role, row["utterance_text"], {
                "utterance_id": int(row["utterance_id"]),
                "timestamp": row["timestamp"],
                "therapist_behaviour": row["main_therapist_behaviour"],
                "client_talk_type": row["client_talk_type"],
            })
            if normalized is not None:
                messages.append(normalized)
        yield finalize_case({
            "case_id": f"annomi:{int(transcript_id):03d}",
            "source": "annomi",
            "language": "en",
            "messages": messages,
            "profile": None,
            "metadata": {
                "mi_quality": first["mi_quality"],
                "topic": first["topic"],
                "video_title": first["video_title"],
                "video_url": first["video_url"],
            },
            "provenance": {
                "source_file": str(path.as_posix()),
                "source_record_id": transcript_id,
                "revision": revision,
            },
        })


def _at(values: Any, index: int) -> Any:
    return values[index] if isinstance(values, list) and index < len(values) else None


def normalize_consistentmi(path: Path, revision: str = "unknown") -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for record_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = []
            for index, (speaker, utterance) in enumerate(zip(row["speakers"], row["utterances"])):
                normalized = _message(speaker, utterance, {
                    "action": _at(row.get("actions"), index),
                    "talk_type": _at(row.get("talk_types"), index),
                    "state": _at(row.get("states"), index),
                    "suggestibility": _at(row.get("suggestibilities"), index),
                })
                if normalized is not None:
                    messages.append(normalized)
            profile = {
                "persona": row.get("Personas"),
                "motivation": row.get("Motivation"),
                "beliefs": row.get("Beliefs"),
                "acceptable_plans": row.get("Acceptable Plans"),
                "behavior": row.get("Behavior"),
            }
            yield finalize_case({
                "case_id": f"consistentmi:{record_index:03d}",
                "source": "consistentmi",
                "language": "en",
                "messages": messages,
                "profile": profile,
                "metadata": {
                    "topic": row.get("topic"),
                    "mi_quality": row.get("quality"),
                    "upstream_idx": row.get("idx"),
                },
                "provenance": {
                    "source_file": str(path.as_posix()),
                    "source_record_id": str(record_index),
                    "revision": revision,
                    "derived_from": f"annomi:{int(row['idx']):03d}",
                },
            })


def normalize_eeyore(path: Path, revision: str = "unknown") -> Iterator[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    role_map = {"user": "counselor", "assistant": "client"}
    for record_index, row in enumerate(records):
        systems = [m.get("content", "").strip() for m in row["conversation"] if m.get("role") == "system"]
        messages = [
            normalized
            for raw in row["conversation"]
            if raw.get("role") in role_map
            if (normalized := _message(role_map[raw["role"]], raw.get("content"))) is not None
        ]
        source_id = row.get("id_source", record_index)
        profile = {
            "structured_profile": row.get("profile"),
            "cognitive_profile": row.get("cognitive profile"),
            "system_profile_prompt": "\n\n".join(systems),
        }
        yield finalize_case({
            "case_id": f"eeyore_public:{record_index:03d}",
            "source": "eeyore_public",
            "language": "en",
            "messages": messages,
            "profile": profile,
            "metadata": {"upstream_source": row.get("source")},
            "provenance": {
                "source_file": str(path.as_posix()),
                "source_record_id": str(source_id),
                "revision": revision,
                "access_scope": "public GitHub evaluation subset",
            },
        })


def normalize_eeyore_full(
    path: Path,
    revision: str = "unknown",
    source_file: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Normalize exported Eeyore profile rows and drop exact dialogue duplicates."""
    seen_dialogues: set[str] = set()
    role_map = {"user": "counselor", "assistant": "client"}
    with path.open(encoding="utf-8") as handle:
        for record_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = [
                normalized
                for raw in row["messages"]
                if raw.get("role") in role_map
                if (normalized := _message(role_map[raw["role"]], raw.get("content"))) is not None
            ]
            fingerprint = hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if fingerprint in seen_dialogues:
                continue
            seen_dialogues.add(fingerprint)
            raw_profile = row.get("profile")
            try:
                profile = json.loads(raw_profile) if isinstance(raw_profile, str) else raw_profile
            except json.JSONDecodeError:
                profile = {"raw_profile": raw_profile, "parse_warning": True}
            yield finalize_case({
                "case_id": f"eeyore_full:{record_index:04d}",
                "source": "eeyore_full",
                "language": "en",
                "messages": messages,
                "profile": profile,
                "metadata": {
                    "upstream_source": row.get("source"),
                    "upstream_id_source": row.get("id_source"),
                    "dialogue_sha256": fingerprint,
                },
                "provenance": {
                    "source_file": source_file or str(path.as_posix()),
                    "source_record_id": str(record_index),
                    "revision": revision,
                    "export_file": str(path.as_posix()),
                    "deduplication": "first row retained for exact normalized-message hash",
                },
            })
