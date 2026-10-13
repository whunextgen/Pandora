from __future__ import annotations


import argparse


import hashlib


import json


import re


import sys


from concurrent.futures import ThreadPoolExecutor, as_completed


from datetime import date


from pathlib import Path


from typing import Any, Literal


from pydantic import BaseModel, ConfigDict, Field, field_validator


ROOT = Path(__file__).resolve().parents[1]


PROJECT_ROOT = ROOT


DEFAULT_CASE_ROOT = ROOT / "data"


PROMPT_VERSION = "2026-09-14-client-grounded-sparse-autobiographical-memory-v6-source-language"


PHASES = ("early", "middle", "late")


FactKind = Literal["event", "experience", "belief", "relation", "coping", "need", "goal"]


DisclosureLevel = Literal[1, 2, 3]


class GroundedFactDraft(BaseModel):
    """One minimally self-contained autobiographical memory explicitly supported by client speech."""

    model_config = ConfigDict(extra="ignore")

    fact_kind: FactKind
    disclosure_level: DisclosureLevel
    content: str
    source_client_turns: list[int] = Field(min_length=1, max_length=4)
    rationale: str = ""

    @field_validator("source_client_turns", mode="before")
    @classmethod
    def normalize_source_client_turns(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return list(dict.fromkeys(value))[:4]

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        cleaned = " ".join(str(value or "").split())
        if not cleaned:
            raise ValueError("content must be non-empty")
        if len(cleaned) > 160:
            raise ValueError("content must stay concise and minimally self-contained")
        if any("\u4e00" <= c <= "\u9fff" for c in cleaned):
            valid_first_person = (
                cleaned.startswith("我")
                and "来访者" not in cleaned
                and "client" not in cleaned.lower()
            )
        else:
            valid_first_person = bool(re.match(r"^(?:I\b|My\b)", cleaned)) and not re.search(
                r"\b(?:the\s+)?client\b", cleaned, flags=re.IGNORECASE
            )
        if not valid_first_person:
            raise ValueError(
                f"first-person format error: {cleaned!r}. "
                "Chinese content must start with 我 and must not use 来访者 as its subject; "
                "English content must start with I or My and contain no client label."
            )
        return cleaned


class GroundedFactSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # No quota and no hard maximum: return only memory-worthy facts supported by the dialogue.
    memories: list[GroundedFactDraft] = Field(default_factory=list)
    summary_rationale: str = ""


class Prompt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: str
    user: str


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_info_dir(case_dir: Path) -> Path:
    organized = case_dir / "1case_info"
    if (organized / "dialog.json").is_file():
        return organized
    return case_dir


def split_dialogue_phases(dialog: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if len(dialog) < 12:
        raise ValueError("dialogue is too short for three-phase extraction")

    pairs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for turn in dialog:
        current.append(turn)
        if turn.get("speaker") == "client":
            pairs.append(current)
            current = []

    if current:
        if not pairs:
            raise ValueError("dialogue contains no client-completed exchange")
        pairs[-1].extend(current)

    if len(pairs) < 3:
        raise ValueError("dialogue has too few client-completed exchanges")

    chunks: dict[str, list[dict[str, Any]]] = {}
    for position, phase in enumerate(PHASES):
        start = position * len(pairs) // 3
        end = (position + 1) * len(pairs) // 3
        chunks[phase] = [turn for pair in pairs[start:end] for turn in pair]
    return chunks


def dialogue_text(turns: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"turn={t.get('turn_id')}; speaker={t.get('speaker')}; text={clean_text(t.get('utterance'))}"
        for t in turns
        if clean_text(t.get("utterance"))
    )


def dialogue_language(turns: list[dict[str, Any]]) -> Literal["Chinese", "English"]:
    text = " ".join(clean_text(turn.get("utterance")) for turn in turns)
    cjk_count = sum("\u4e00" <= char <= "\u9fff" for char in text)
    latin_count = sum(char.isascii() and char.isalpha() for char in text)
    return "English" if latin_count > cjk_count else "Chinese"


def build_prompt(*, dataset_id: str, phase: str, turns: list[dict[str, Any]]) -> Prompt:
    language = dialogue_language(turns)
    valid_client_turn_ids = [
        int(t["turn_id"])
        for t in turns
        if t.get("speaker") == "client" and isinstance(t.get("turn_id"), int)
    ]

    return Prompt(
        system=(
            "You extract a SPARSE, auditable psychotherapy CLIENT autobiographical memory bank from a source dialogue. "
            "The memory bank is NOT a transcript summary and should skip most conversational turns. "
            "Each card must be a minimally self-contained, person-specific fact or theme explicitly stated or explicitly endorsed by the CLIENT. "
            f"The source dialogue is {language}. Write every memory in that same language; never translate it. "
            "Write every memory in the CLIENT\'S FIRST-PERSON VOICE. English content must start with I or My; "
            "Chinese content must start with 我, including relationship facts: '我觉得……', '我有时候觉得……', '我男友比我大三岁……'. "
            "Do not use 她/来访者/client as the CLIENT subject. Pronouns for another explicitly identified person are allowed. "
            "In Chinese content, prefer explicit relationships such as 我妈妈 when the referent would otherwise be ambiguous. "
            "For example: '我最近没和妈妈吵架，她也少打电话' must be '我最近没和我妈妈吵架，我妈妈也少打电话'. "
            "Rewrite perspective only: preserve the original people, facts, uncertainty, and meaning; do not omit the first-person subject. "
            "Prefer omission over weak inclusion. Do not infer psychological mechanisms, latent traits, schemas, diagnoses, causal explanations, or therapist formulations."
        ),
        user=(
            f"dataset_id: {dataset_id}\n"
            f"source_dialogue_phase: {phase}\n"
            f"prompt_version: {PROMPT_VERSION}\n\n"
            f"Valid source_client_turns: {valid_client_turn_ids}\n\n"
            "Dialogue:\n"
            f"{dialogue_text(turns)}\n\n"
            "Extract only DISTINCT, MEMORY-WORTHY CLIENT FACTS/THEMES that would materially help simulate this same person in later turns. "
            "The goal is a sparse autobiographical memory bank, NOT coverage of the dialogue. It is normal and desirable for many client turns to produce no card. "
            "Return only facts directly supported by client speech. There is NO target number of cards and NO phase quota. When in doubt, OMIT.\n\n"
            "A candidate should be kept only if it passes ALL of these persistence tests:\n"
            "1. PERSON-SPECIFIC: it distinguishes this client or preserves important personal history/context, rather than expressing a generic wish or conversational response.\n"
            "2. CROSS-TURN USEFUL: knowing it later would materially improve continuity, consistency, or person-specific simulation.\n"
            "3. STANDALONE: it remains meaningful outside the immediately preceding counselor turn; avoid vague references such as 'this', 'that', 'such a reminder' without a self-contained client fact.\n"
            "4. CLIENT-OWNED: it originates from the client's own history/view/experience, not primarily from a counselor suggestion that the client merely accepts.\n\n"
            "ALLOWED fact kinds:\n"
            "- event: a concrete event or durable life circumstance explicitly described by the client\n"
            "- experience: an emotionally meaningful or clinically salient subjective experience explicitly described by the client\n"
            "- belief: a thought, expectation, attribution, value, or self-evaluation explicitly verbalized by the client\n"
            "- relation: a concrete relationship fact or interaction pattern explicitly described by the client\n"
            "- coping: an established or recurring behavior/avoidance/communication strategy explicitly described by the client; NOT a newly accepted counselor suggestion\n"
            "- need: a specific, person-relevant emotional/interpersonal need explicitly stated by the client\n"
            "- goal: a specific, person-relevant goal that reflects the client's own concern; NOT generic 'I want to get better/happy' or a counselor-induced commitment\n\n"
            "DISCLOSURE LEVEL describes only how difficult/private the EXPLICIT FACT would be for this client to disclose; it is NOT a clinical depth label:\n"
            "- L1: ordinary contextual facts, concrete events, relationship facts, everyday behaviors, or low-vulnerability preferences/coping actions. "
            "A statement does NOT become L2 merely because it mentions emotion or coping.\n"
            "- L2: clearly personal and meaningfully vulnerable facts, such as specific fears, dependency, relationship needs, avoidance driven by conflict, "
            "painful emotions, or negative self-evaluations that reveal more than ordinary context.\n"
            "- L3: unusually difficult and highly vulnerable explicit facts, such as severe shame or worthlessness, 'I am unlovable/defective', "
            "'I cannot escape this', 'I cannot keep going', intense abandonment fear, or comparably painful self-revelation. "
            "Use L3 sparingly and only when the client directly verbalizes the full proposition.\n\n"
            "HARD EXCLUSIONS -- DO NOT STORE THESE AS MEMORIES:\n"
            "- Counselor-induced commitments or uptake after advice: e.g. 'I will try', 'I will remember', 'I will work on it', 'I will communicate when calm', "
            "'I need this reminder', 'writing a letter sounds like a good idea', 'I will try to fill these gaps', 'this is a gradual process', or similar statements whose main function is accepting, paraphrasing, or adopting the counselor's suggestion. "
            "Explicit Chinese negative examples include: '写信听起来是个好办法，可以慢慢整理好情绪和语言', '我会尽力去填平这些坑洼', and '这是一个慢慢进步的过程'.\n"
            "- Generic aspirations with little person-specific information: e.g. merely wanting to be happy, become better, improve, be confident, or solve the problem. "
            "Keep a goal only when its content is specific to this person's recurring conflict or life context.\n"
            "- Session-local acknowledgements, meta-reflections, or uncertainty: e.g. 'I am trying to understand this', 'I don't know the answer', "
            "'maybe you're right', 'I will think about it', 'I need more time', unless the utterance itself contains a durable person-specific fact.\n"
            "- Short-lived reactions that only summarize the immediate exchange and do not characterize the client beyond that moment.\n"
            "- Vague deictic statements whose meaning depends on the counselor's words ('this', 'that', 'such a reminder') unless the client explicitly states the referent in the same cited turn(s).\n\n"
            "KEEP preferentially:\n"
            "- concrete personal/family history and durable circumstances;\n"
            "- durable relationship context and recurring interactions explicitly described by the client;\n"
            "- specific recurring fears, expectations, dependencies, conflicts, and avoidance/coping patterns explicitly described by the client;\n"
            "- distinctive self-evaluations, beliefs, values, and highly vulnerable admissions explicitly verbalized by the client.\n\n"
            "MINIMAL SELF-CONTAINMENT AND DEDUPLICATION:\n"
            "- One card should represent one distinctive autobiographical fact/theme, not necessarily one sentence fragment.\n"
            "- Deduplicate across the ENTIRE phase, not only adjacent turns. If the client explicitly repeats or elaborates the SAME autobiographical theme at several points in the phase, MERGE those repetitions into one concise card rather than emitting multiple paraphrases.\n"
            "- Merge only when every cited turn explicitly supports the same underlying proposition; do not infer a broader trait or mechanism from several examples.\n"
            "- If two candidates are semantically overlapping, keep the more specific and informative wording. Do not emit paraphrase duplicates.\n"
            "- Split only when the statements would independently matter for later simulation.\n"
            "- Prefer one supporting client turn; use multiple turns only when the same fact/theme is explicitly repeated, completed, or clarified. Keep the evidence set small and directly entailing.\n"
            "- source_client_turns must contain 1-4 distinct client turn ids; never return more than 4.\n"
            "- The entire card must be entailed by the cited client turn(s), not merely plausible from surrounding dialogue.\n"
            "- Keep the wording close to the client's own framing. Do not summarize across many turns into a therapist-style recurring pattern.\n"
            "- Do NOT infer attachment style, defense mechanism, schema, unconscious motive, learned helplessness, reinforcement mechanism, personality structure, core psychological mechanism, or therapist-side causal formulation.\n"
            "- Do NOT transform repeated examples into a generalized trait/pattern unless the client explicitly states that generalization.\n"
            "- Preserve modality and uncertainty ('maybe', 'sometimes', 'I worry', 'I feel'). Do not strengthen causality.\n"
            "- Counselor interpretations or suggestions are not client facts merely because the client agrees.\n"
            "- If a phase contains no explicit L3 fact, return no L3 card. If only a few facts pass the persistence tests, return only those few.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "memories": [{\n'
            '    "fact_kind": "event|experience|belief|relation|coping|need|goal",\n'
            '    "disclosure_level": 1,\n'
            f'    "content": "{language}, first-person, one topic, close to the client\'s wording",\n'
            '    "source_client_turns": [12],\n'
            '    "rationale": "brief grounding note; no clinical interpretation"\n'
            "  }],\n"
            '  "summary_rationale": "brief extraction note"\n'
            "}\n"
        ),
    )


def validate_summary(
    summary: GroundedFactSummary,
    *,
    phase: str,
    turns: list[dict[str, Any]],
) -> None:
    language = dialogue_language(turns)
    client_turn_ids = {
        int(t["turn_id"])
        for t in turns
        if t.get("speaker") == "client" and isinstance(t.get("turn_id"), int)
    }
    for memory in summary.memories:
        GroundedFactDraft.validate_content(memory.content)
        has_cjk = any("\u4e00" <= char <= "\u9fff" for char in memory.content)
        if (language == "English" and has_cjk) or (language == "Chinese" and not has_cjk):
            raise ValueError(f"{phase}: memory content must preserve source language ({language})")
        memory.source_client_turns = [
            turn_id for turn_id in memory.source_client_turns if turn_id in client_turn_ids
        ]
        if not memory.source_client_turns:
            raise ValueError(f"{phase}: memory cites non-client or out-of-phase turns")


def compute_summary(
    *,
    provider: object,
    dataset_id: str,
    phase: str,
    turns: list[dict[str, Any]],
    attempts: int = 4,
) -> GroundedFactSummary:
    parser = getattr(provider, "_parse_with_openai", None)
    if not callable(parser):
        raise RuntimeError("requires a structured LLM provider")

    last_error: Exception | None = None
    language = dialogue_language(turns)
    perspective_rule = (
        "Every content must start with I or My; do not use client as the subject. "
        if language == "English"
        else "Every content must start with 我; do not use 她/来访者 as the client subject. "
    )
    for _ in range(attempts):
        try:
            prompt = build_prompt(dataset_id=dataset_id, phase=phase, turns=turns)
            if last_error is not None:
                prompt.user += (
                    "\nPrevious output failed validation. Correct the output before returning it: "
                    f"{last_error}\n{perspective_rule}"
                    f"The source dialogue is {language}; keep every memory in {language} and do not translate. "
                    "Preserve facts and relationships; do not drop memories merely to fix their grammatical perspective."
                )
            summary = parser(
                prompt=prompt,
                schema=GroundedFactSummary,
            )
            validate_summary(summary, phase=phase, turns=turns)
            return summary
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"{dataset_id} {phase}: failed after {attempts} attempts; "
        f"last_error={type(last_error).__name__}: {last_error}"
    ) from last_error


def build_memory_bank(
    *, filesystem_id: str, summaries: dict[str, GroundedFactSummary]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the runtime bank while keeping all psychological content fact-grounded.

    Runtime compatibility notes:
    - ``kind`` is always ``surface`` because every card is explicitly client-accessible.
    - ``disclosure_level`` is independent and captures privacy/vulnerability only.
    - ``activation_threshold`` is fixed to 0.0 solely because the current MemoryItem schema
      requires the field; the full-memory-oracle runtime does not use it as a gate.
    """

    memories: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    seen: set[str] = set()

    for phase_index, phase in enumerate(PHASES, start=1):
        for local_index, draft in enumerate(summaries[phase].memories, start=1):
            normalized = re.sub(r"\W+", "", draft.content).lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)

            memory_id = (
                f"{filesystem_id}-fact-{phase}-l{draft.disclosure_level}-{local_index:02d}"
            )
            evidence = [f"dialog.client_turn[{tid}]" for tid in draft.source_client_turns]

            # Keep the runtime card intentionally small.  No mechanism/core-belief tags,
            # inferred clinical relevance, or learned thresholds are stored here.
            memories.append(
                {
                    "memory_id": memory_id,
                    "kind": "surface",
                    "content": draft.content,
                    "accessibility": "surface",
                    "type": draft.fact_kind,
                    "memory_type": "client_grounded_autobiographical_fact",
                    "session_id": 1,
                    "source_session_id": 1,
                    "visibility_start_session": 1,
                    "available_from_session": 1,
                    "visibility_start_turn": 0,
                    "source": f"source_dialogue_phase.{phase}",
                    "source_path": f"source_dialogue_phase.{phase}",
                    "client_visibility": "directly_accessible",
                    "expression_permission": "can_say",
                    "verbatim_quote": False,
                    "disclosure_level": int(draft.disclosure_level),
                    "activation_threshold": 0.0,
                    "salience": 0.5,
                    "source_turn": min(draft.source_client_turns),
                    "evidence": evidence,
                }
            )

            audit.append(
                {
                    "memory_id": memory_id,
                    "fact_kind": draft.fact_kind,
                    "disclosure_level": int(draft.disclosure_level),
                    "source_phase": phase,
                    "source_phase_index": phase_index,
                    "source_client_turns": draft.source_client_turns,
                    "rationale": clean_text(draft.rationale),
                }
            )

    if not memories:
        raise ValueError("empty memory bank")
    return memories, audit


def build_disclosure_order(
    *, dataset_id: str, memories: list[dict[str, Any]], audit: list[dict[str, Any]]
) -> dict[str, Any]:
    audit_by_id = {a["memory_id"]: a for a in audit}
    ordered = sorted(
        memories,
        key=lambda m: (
            audit_by_id[m["memory_id"]]["source_phase_index"],
            m["source_turn"],
            m["disclosure_level"],
            m["memory_id"],
        ),
    )
    return {
        "schema_version": "4.0",
        "dataset_id": dataset_id,
        "ordering_policy": {
            "primary_key": "source dialogue phase",
            "secondary_key": "earliest supporting client turn",
            "runtime_note": (
                "All cards are explicit client-grounded facts. disclosure_level encodes "
                "privacy/vulnerability for progressive release, not inferred psychological depth."
            ),
        },
        "sequence": [
            {
                "order": i,
                "memory_id": m["memory_id"],
                "layer": f"L{m['disclosure_level']}",
                "disclosure_level": m["disclosure_level"],
                "source_phase": audit_by_id[m["memory_id"]]["source_phase"],
                "source_client_turns": audit_by_id[m["memory_id"]]["source_client_turns"],
            }
            for i, m in enumerate(ordered, start=1)
        ],
    }


def generate_case(
    *, case_dir: Path, provider: object, provider_model: str, overwrite: bool
) -> dict[str, Any]:
    case_dir = case_info_dir(case_dir)
    filesystem_id = case_dir.parent.name if case_dir.name == "1case_info" else case_dir.name

    output_path = case_dir / "memory_bank.json"
    if output_path.exists() and not overwrite:
        return {
            "dataset_id": filesystem_id,
            "status": "skipped_existing",
            "output": str(output_path),
        }

    dialog_path = case_dir / "dialog.json"
    dialog_payload = read_json(dialog_path)
    dialog = dialog_payload.get("dialog")
    if not isinstance(dialog, list) or not dialog:
        raise ValueError(f"{case_dir}: dialog.json.dialog missing")

    dataset_id = dialog_payload.get("dataset_id") or dialog_payload.get("case_id") or filesystem_id
    phases = split_dialogue_phases(dialog)
    summaries = {
        phase: compute_summary(
            provider=provider,
            dataset_id=dataset_id,
            phase=phase,
            turns=phases[phase],
        )
        for phase in PHASES
    }

    memories, audit = build_memory_bank(filesystem_id=filesystem_id, summaries=summaries)
    order = build_disclosure_order(dataset_id=dataset_id, memories=memories, audit=audit)

    # Canonical files consumed directly by the experiment runner.
    write_json(case_dir / "memory_bank.json", memories)
    write_json(case_dir / "memory_disclosure_order.json", order)

    # Separate audit artifacts retain provenance without polluting runtime memory cards.
    write_json(
        case_dir / "memory_grounding_audit.json",
        {
            "schema_version": "2.0",
            "dataset_id": dataset_id,
            "prompt_version": PROMPT_VERSION,
            "memory_audit": audit,
            "phase_summaries": {
                phase: summaries[phase].model_dump(mode="json") for phase in PHASES
            },
        },
    )
    write_json(
        case_dir / "memory_grounding_metadata.json",
        {
            "schema_version": "2.0",
            "dataset_id": dataset_id,
            "mode": "client_grounded_sparse_autobiographical_memory_extraction",
            "provider_model": provider_model,
            "prompt_version": PROMPT_VERSION,
            "generated_at": date.today().isoformat(),
            "memory_count": len(memories),
            "semantic_contract": {
                "runtime_memory": "sparse, minimally self-contained autobiographical facts/themes explicitly stated or endorsed by the client",
                "disclosure_level": "privacy/vulnerability of an explicit fact, not clinical depth",
                "clinical_formulation": "excluded from memory_bank.json",
                "activation_threshold": "compatibility field fixed to 0.0 and unused by full_memory_oracle",
            },
            "source_hash": sha256_file(dialog_path),
        },
    )

    return {
        "dataset_id": dataset_id,
        "status": "generated",
        "memory_count": len(memories),
        "output": str(output_path),
    }
