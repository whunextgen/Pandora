from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "selected_cases.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "selected_cases_by_id"

LAYER_CONFIG: dict[str, dict[str, Any]] = {
    "L1": {
        "disclosure_level": 1,
        "kind": "surface",
        "accessibility": "surface",
        "client_visibility": "directly_accessible",
        "expression_permission": "can_say",
        "activation_threshold": 0.20,
        "definition": "来访者清楚知道并可直接陈述的事实、症状、事件和感受。",
    },
    "L2": {
        "disclosure_level": 2,
        "kind": "mechanism",
        "accessibility": "mechanism",
        "client_visibility": "partially_accessible",
        "expression_permission": "indirect_only",
        "activation_threshold": 0.42,
        "definition": "来访者有所感知，但表达仍试探、不完整的模式、冲突、应对倾向或感受。",
    },
    "L3": {
        "disclosure_level": 3,
        "kind": "latent_clinical",
        "accessibility": "latent_clinical",
        "client_visibility": "not_directly_accessible",
        "expression_permission": "latent_influence_only",
        "activation_threshold": 0.60,
        "definition": "经探询或反思逐渐浮现的早期经历、核心信念、依恋或身份冲突及因果联系。",
    },
}


def load_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def filesystem_id(dataset_id: str) -> str:
    """Return a reversible, Windows-safe directory name for a canonical case id."""
    safe = dataset_id.replace(":", "__")
    safe = re.sub(r'[<>"/\\|?*\x00-\x1f]', "_", safe).rstrip(" .")
    if not safe:
        raise ValueError(f"dataset_id cannot be converted to a directory name: {dataset_id!r}")
    return safe


def validate_case(case: dict[str, Any]) -> None:
    dataset_id = case.get("case_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("selected case is missing a non-empty case_id")
    messages = case.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{dataset_id}: messages must be a non-empty list")
    for position, message in enumerate(messages):
        if message.get("index") != position:
            raise ValueError(f"{dataset_id}: message index mismatch at position {position}")
    selection = case.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"{dataset_id}: selection metadata is missing")
    evidence = selection.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"{dataset_id}: selection.evidence is missing")
    for layer in LAYER_CONFIG:
        items = evidence.get(layer)
        if not isinstance(items, list) or not items:
            raise ValueError(f"{dataset_id}: {layer} evidence must be non-empty")
        for item in items:
            index = item.get("message_index")
            if not isinstance(index, int) or not 0 <= index < len(messages):
                raise ValueError(f"{dataset_id}: invalid {layer} message_index {index!r}")
            message = messages[index]
            if message.get("role") != "client":
                raise ValueError(f"{dataset_id}: {layer} evidence {index} is not client speech")
            if item.get("quote") != message.get("content"):
                raise ValueError(f"{dataset_id}: {layer} quote does not match message {index}")


def eliciting_counselor_indices(selection: dict[str, Any], client_index: int) -> list[int]:
    indices: list[int] = []
    for chain in selection.get("elicitation_chains", []):
        client_indices = sorted(chain.get("client_indices", []))
        if client_index not in client_indices:
            continue
        client_position = client_indices.index(client_index)
        previous_client_index = client_indices[client_position - 1] if client_position else -1
        indices.extend(
            index
            for index in chain.get("counselor_indices", [])
            if previous_client_index < index < client_index
        )
    return sorted(set(indices))


def build_memory_bank(case: dict[str, Any], folder_id: str) -> list[dict[str, Any]]:
    selection = case["selection"]
    memories: list[dict[str, Any]] = []
    for layer in LAYER_CONFIG:
        config = LAYER_CONFIG[layer]
        for ordinal, evidence in enumerate(selection["evidence"][layer], start=1):
            source_turn = evidence["message_index"]
            memory_id = f"{folder_id}-{layer.lower()}-{ordinal:03d}"
            counselor_indices = eliciting_counselor_indices(selection, source_turn)
            evidence_paths = [f"original_dialog.messages[{source_turn}]"]
            evidence_paths.extend(
                f"original_dialog.messages[{index}] (counselor elicitation)"
                for index in counselor_indices
            )
            memories.append({
                "memory_id": memory_id,
                "kind": config["kind"],
                "content": evidence["quote"],
                "accessibility": config["accessibility"],
                "type": config["kind"],
                "memory_type": "three_layer_disclosure_evidence",
                "source": f"selection.evidence.{layer}",
                "source_path": f"selection.evidence.{layer}[{ordinal - 1}]",
                "client_visibility": config["client_visibility"],
                "expression_permission": config["expression_permission"],
                "clinical_relevance": "review_required",
                "verbatim_quote": True,
                "disclosure_level": config["disclosure_level"],
                "activation_threshold": config["activation_threshold"],
                "salience": 0.50,
                "source_turn": source_turn,
                "evidence": evidence_paths,
            })
    return memories


def build_disclosure_order(
    case: dict[str, Any], memories: list[dict[str, Any]]
) -> dict[str, Any]:
    selection = case["selection"]
    ordered = sorted(
        memories,
        key=lambda item: (item["source_turn"], item["disclosure_level"], item["memory_id"]),
    )
    sequence = []
    for order, memory in enumerate(ordered, start=1):
        source_turn = memory["source_turn"]
        sequence.append({
            "order": order,
            "memory_id": memory["memory_id"],
            "layer": f"L{memory['disclosure_level']}",
            "disclosure_level": memory["disclosure_level"],
            "source_message_index": source_turn,
            "elicited_by_counselor_indices": eliciting_counselor_indices(selection, source_turn),
        })
    index_layers: dict[int, list[str]] = {}
    for item in sequence:
        index_layers.setdefault(item["source_message_index"], []).append(item["layer"])
    overlaps = [
        {"source_message_index": index, "layers": layers}
        for index, layers in sorted(index_layers.items())
        if len(layers) > 1
    ]
    return {
        "schema_version": "1.0",
        "dataset_id": case["case_id"],
        "ordering_policy": {
            "primary_key": "source_message_index ascending",
            "tie_breaker": "disclosure_level ascending, then memory_id",
            "meaning": "Replays the order in which the selected evidence appeared in the source dialogue.",
            "runtime_note": (
                "This file records the gold disclosure trajectory. A simulator should still require "
                "appropriate counselor elicitation and readiness before exposing L2/L3 content."
            ),
        },
        "layer_definitions": {
            layer: config["definition"] for layer, config in LAYER_CONFIG.items()
        },
        "sequence": sequence,
        "layer_groups": {
            layer: [
                memory["memory_id"]
                for memory in memories
                if memory["disclosure_level"] == config["disclosure_level"]
            ]
            for layer, config in LAYER_CONFIG.items()
        },
        "elicitation_chains": selection.get("elicitation_chains", []),
        "overlapping_evidence_indices": overlaps,
    }


def build_client_info(case: dict[str, Any]) -> dict[str, Any]:
    selection = case["selection"]
    evidence = selection["evidence"]
    source_profile = case.get("profile")
    parsed = {
        "编号": case["case_id"],
        "主诉": {
            "主诉类型": "原数据未单独标注",
            "主诉内容": [item["quote"] for item in evidence["L1"]],
        },
        "来访者个人信息": (
            source_profile if source_profile is not None else "原数据 profile 字段为空，不推断人口学信息"
        ),
        "背景信息": selection.get("rationale"),
        "三层记忆证据": {
            layer: [
                {
                    "message_index": item["message_index"],
                    "quote": item["quote"],
                }
                for item in evidence[layer]
            ]
            for layer in LAYER_CONFIG
        },
        "记忆披露特征": {
            "scores": selection.get("scores", {}),
            "score_total": selection.get("score_total"),
            "profile_leakage_warning": selection.get("profile_leakage_warning", False),
            "severity_only_warning": selection.get("severity_only_warning", False),
            "review_status": selection.get("status"),
        },
    }
    return {
        "source_file": "data/processed/selected_cases.jsonl",
        "format": "pandora_three_layer_evidence_profile",
        "profile_provenance": {
            "source_profile_present": source_profile is not None,
            "derivation": "selection evidence and rationale only; no demographic or clinical facts invented",
            "reviewer": selection.get("reviewer"),
            "review_version": selection.get("review_version"),
        },
        "client_info": {"parsed": parsed},
    }


def build_dialog(case: dict[str, Any]) -> dict[str, Any]:
    """Adapt the canonical case to the dialog.json shape used by therapist distillation."""
    system_prompt = case.get("metadata", {}).get("system_prompt")
    return {
        "schema_version": "1.0",
        "case_id": case["case_id"],
        "dataset_id": case["case_id"],
        "source": case.get("source"),
        "language": case.get("language"),
        "system_prompt": system_prompt,
        "dialog": [
            {
                "session_id": 1,
                "turn_id": message["index"],
                "source_message_index": message["index"],
                "speaker": "therapist" if message["role"] == "counselor" else "client",
                "utterance": message["content"],
                "annotations": message.get("annotations"),
            }
            for message in case["messages"]
        ],
        "provenance": case.get("provenance", {}),
        "adapter_note": (
            "Lossless role/field adapter for the therapist-distillation workflow; "
            "the canonical source remains original_dialog.json."
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_case(
    case: dict[str, Any], *, line_number: int, input_path: Path, output_root: Path
) -> dict[str, Any]:
    validate_case(case)
    dataset_id = case["case_id"]
    folder_id = filesystem_id(dataset_id)
    case_dir = output_root / folder_id
    case_dir.mkdir(parents=True, exist_ok=True)

    memories = build_memory_bank(case, folder_id)
    disclosure_order = build_disclosure_order(case, memories)
    client_info = build_client_info(case)
    initialization_metadata = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "filesystem_id": folder_id,
        "source_case_id": dataset_id,
        "source": case.get("source"),
        "source_record_id": case.get("provenance", {}).get("source_record_id"),
        "source_selected_cases_file": portable_path(input_path),
        "source_line_number": line_number,
        "review_status": case["selection"].get("status"),
        "selection_evidence_policy": {
            "memory_type": "three_layer_disclosure_evidence",
            "memory_content": "verbatim client evidence selected in the source package",
            "layer_mapping": {layer: config["disclosure_level"] for layer, config in LAYER_CONFIG.items()},
            "activation_thresholds": {
                layer: config["activation_threshold"] for layer, config in LAYER_CONFIG.items()
            },
            "cards_file": "selection_evidence_memory_cards.json",
            "order_file": "selection_evidence_order.json",
            "runtime_boundary": (
                "Selection evidence is audit material, not the simulation memory bank. "
                "Run scripts/enrich_simulation_memories.py to create memory_bank.json."
            ),
        },
        "limitations": [
            "The selected case is a provisional model-assisted seed, not a clinical expert gold label.",
            "The source profile is null unless profile_provenance.source_profile_present is true.",
            "Memory cards preserve selected verbatim evidence and must not be treated as independent clinical inference.",
        ],
    }

    payloads = {
        "original_dialog.json": case,
        "dialog.json": build_dialog(case),
        "client_info.json": client_info,
        "selection_evidence_memory_cards.json": memories,
        "selection_evidence_order.json": disclosure_order,
        "initialization_metadata.json": initialization_metadata,
    }
    for filename, payload in payloads.items():
        write_json(case_dir / filename, payload)

    file_manifest = {
        filename: {
            "sha256": sha256(case_dir / filename),
            "bytes": (case_dir / filename).stat().st_size,
        }
        for filename in payloads
    }
    layer_counts = Counter(f"L{memory['disclosure_level']}" for memory in memories)
    manifest = {
        "schema_version": "1.0",
        "dataset_id": dataset_id,
        "filesystem_id": folder_id,
        "files": file_manifest,
        "message_count": len(case["messages"]),
        "selection_evidence_card_count": len(memories),
        "selection_evidence_count_by_layer": dict(sorted(layer_counts.items())),
        "overlapping_evidence_indices": disclosure_order["overlapping_evidence_indices"],
    }
    write_json(case_dir / "manifest.json", manifest)
    return manifest


def build_root_readme(output_root: Path) -> None:
    text = """# 按 dataset_id 展开的三层记忆 case

本目录由 `scripts/materialize_case_folders.py` 从
`data/processed/selected_cases.jsonl` 确定性生成。

每个子目录对应一个已筛选 case，包含：

- `original_dialog.json`：完整保留统一 schema 下的原 case、对话和 selection 标注；
- `dialog.json`：供 therapist distillation 使用的无损适配格式，并保留原始 system prompt；
- `client_info.json`：证据型来访者 profile；源 profile 为空时不补造人口学或临床事实；
- `selection_evidence_memory_cards.json`：筛选时使用的逐字 L1/L2/L3 证据卡，仅供审计；
- `selection_evidence_order.json`：上述筛选证据按原对话 message index 排列的出现顺序和促发链；
- `memory_bank.json`：运行 `scripts/enrich_simulation_memories.py` 后生成的完整、非逐字模拟记忆库；建议每个长对话形成约 18--24 条互不重复、覆盖三层的运行记忆；
- `memory_disclosure_order.json`：完整模拟记忆库的来源阶段顺序与披露层级；
- `initialization_metadata.json`：生成口径、来源和限制；
- `manifest.json`：文件校验值和逐层记忆数量。
- `therapist_distillation/`：运行 `scripts/distill_case_therapists.py` 后生成的 case-specific 咨询师技能包；该目录不由本脚本删除或覆盖。
- `client_distillation/`：可选的 case-specific 来访者技能包；稳定表达只从 `client_info.json` 和 `dialog.json` 蒸馏，运行时再独立接入完整记忆库；该目录不由本脚本删除或覆盖。

Windows 目录名不能包含冒号，因此 canonical dataset id 中的 `:` 以 `__`
表示。例如 `psydial_d4:0561` 的目录名是 `psydial_d4__0561`，原始 ID
始终保存在每个 JSON 文件的 `dataset_id` 或 `case_id` 字段中。

注意：这些 case 仍是模型辅助的 provisional seed，不能直接称为临床专家标注金标准。

推荐生成顺序：先运行 `python scripts/materialize_case_folders.py`，再依次运行
`python scripts/enrich_simulation_memories.py --overwrite` 和
`python scripts/distill_case_therapists.py --overwrite`。

如模型服务暂时不可用，可以对单个 case 提供显式人工审核草稿：
`python scripts/enrich_simulation_memories.py --case-id <id> --reviewed-draft <case内草稿路径> --overwrite`。
该参数不会作为静默自动降级启用。
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize selected PAnDora cases into one directory per dataset id."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_root = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_root.mkdir(parents=True, exist_ok=True)

    manifests = [
        materialize_case(
            case,
            line_number=line_number,
            input_path=input_path,
            output_root=output_root,
        )
        for line_number, case in load_jsonl(input_path)
    ]
    if not manifests:
        raise ValueError(f"no cases found in {input_path}")
    folder_ids = [manifest["filesystem_id"] for manifest in manifests]
    if len(folder_ids) != len(set(folder_ids)):
        raise ValueError("multiple dataset ids map to the same filesystem id")

    build_root_readme(output_root)
    index = {
        "schema_version": "1.0",
        "source_file": portable_path(input_path),
        "case_count": len(manifests),
        "cases": [
            {
                "dataset_id": manifest["dataset_id"],
                "filesystem_id": manifest["filesystem_id"],
                "selection_evidence_card_count": manifest["selection_evidence_card_count"],
                "selection_evidence_count_by_layer": manifest["selection_evidence_count_by_layer"],
            }
            for manifest in manifests
        ],
    }
    write_json(output_root / "index.json", index)
    print(json.dumps(index, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
