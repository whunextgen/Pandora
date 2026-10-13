from __future__ import annotations

from typing import Any, Iterable


SCHEMA_VERSION = "1.0"
ROLES = {"counselor", "client"}


def count_exchanges(messages: Iterable[dict[str, Any]]) -> int:
    """Count complete counselor/client runs, regardless of who speaks first."""
    runs: list[str] = []
    for message in messages:
        role = message.get("role")
        if role not in ROLES:
            continue
        if not runs or runs[-1] != role:
            runs.append(role)
    return len(runs) // 2


def case_stats(messages: list[dict[str, Any]]) -> dict[str, Any]:
    client = [m for m in messages if m["role"] == "client"]
    counselor = [m for m in messages if m["role"] == "counselor"]
    transitions = sum(
        messages[index - 1]["role"] != messages[index]["role"]
        for index in range(1, len(messages))
    )
    return {
        "message_count": len(messages),
        "exchange_count": count_exchanges(messages),
        "client_message_count": len(client),
        "counselor_message_count": len(counselor),
        "client_character_count": sum(len(m["content"]) for m in client),
        "counselor_character_count": sum(len(m["content"]) for m in counselor),
        "alternation_ratio": transitions / max(1, len(messages) - 1),
    }


def finalize_case(case: dict[str, Any]) -> dict[str, Any]:
    for index, message in enumerate(case["messages"]):
        message["index"] = index
    case["schema_version"] = SCHEMA_VERSION
    case["stats"] = case_stats(case["messages"])
    validate_case(case)
    return case


def validate_case(case: dict[str, Any]) -> None:
    required = {
        "schema_version", "case_id", "source", "language", "messages",
        "profile", "metadata", "provenance",
    }
    missing = sorted(required - set(case))
    if missing:
        raise ValueError(f"missing case fields: {', '.join(missing)}")
    if not case["case_id"] or not case["source"]:
        raise ValueError("case_id and source must be non-empty")
    if not isinstance(case["messages"], list) or not case["messages"]:
        raise ValueError("messages must be a non-empty list")
    for index, message in enumerate(case["messages"]):
        if message.get("index") != index:
            raise ValueError(f"message index mismatch at {index}")
        if message.get("role") not in ROLES:
            raise ValueError(f"invalid role at {index}: {message.get('role')}")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"empty message content at {index}")
    provenance = case["provenance"]
    if not provenance.get("source_file") or provenance.get("source_record_id") is None:
        raise ValueError("provenance requires source_file and source_record_id")

