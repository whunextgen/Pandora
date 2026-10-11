from __future__ import annotations


import json


from typing import Any


from pandora.schemas import ClientResponse, LatentPsychologicalState


MAX_DELTA = {
    "distress": 0.06,
    "defensiveness": 0.05,
    "perceived_safety": 0.04,
    "trust": 0.04,
    "emotional_arousal": 0.06,
    "avoidance_tendency": 0.05,
    "safety_risk": 0.03,
    "therapeutic_alliance": 0.03,
    "disclosure_readiness": 0.04,
    "self_awareness_level": 0.04,
    "insight_level": 0.03,
    "mentalization_level": 0.03,
    "core_belief_activation": 0.05,
    "problem_clarity": 0.04,
}


TYPE_DELTA_SCALE = {
    "logistics": 0.10,
    "factual_question": 0.40,
    "reflective_statement": 0.50,
    "challenge_or_confrontation": 0.80,
    "invalidation": 1.00,
}


SLOW_FIELD_MAX_DELTA = {
    "trust": 0.035,
    "perceived_safety": 0.035,
    "therapeutic_alliance": 0.025,
}


SLOW_FIELDS = set(SLOW_FIELD_MAX_DELTA)


SOFT_BOUNDS = {
    "distress": (0.00, 0.92),
    "defensiveness": (0.00, 0.90),
    "emotional_arousal": (0.00, 0.88),
    "avoidance_tendency": (0.00, 0.85),
    "trust": (0.10, 0.90),
    "perceived_safety": (0.05, 0.85),
    "therapeutic_alliance": (0.05, 0.85),
    "safety_risk": (0.00, 0.35),
}


NUMERIC_STATE_FIELDS = {
    name for name, field in LatentPsychologicalState.model_fields.items() if field.annotation is float
}


AROUSAL_FIELDS = {
    "distress",
    "defensiveness",
    "emotional_arousal",
    "avoidance_tendency",
    "core_belief_activation",
}


SAFETY_FIELDS = {"perceived_safety", "trust", "disclosure_readiness"}


REFLECTIVE_FIELDS = {"self_awareness_level", "insight_level", "mentalization_level", "problem_clarity"}


def _render_json(value: object) -> str:
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json(indent=2)  # type: ignore[attr-defined]
    if isinstance(value, list):
        materialized: list[object] = []
        for item in value:
            if hasattr(item, "model_dump"):
                materialized.append(item.model_dump(mode="json"))  # type: ignore[attr-defined]
            else:
                materialized.append(item)
        return json.dumps(materialized, ensure_ascii=False, indent=2)
    return json.dumps(value, ensure_ascii=False, indent=2)


def _normalize_utterance_type(value: str | None) -> str:
    normalized = (value or "factual_question").strip().lower()
    if normalized not in TYPE_DELTA_SCALE:
        return "factual_question"
    return normalized


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _base_max_delta(field: str) -> float:
    max_delta = MAX_DELTA.get(field, 0.05)
    if field in SLOW_FIELD_MAX_DELTA:
        max_delta = min(max_delta, SLOW_FIELD_MAX_DELTA[field])
    return max_delta


def _apply_soft_bounds(field: str, value: float) -> float:
    lower, upper = SOFT_BOUNDS.get(field, (0.0, 1.0))
    return _clip(value, lower, upper)


def _apply_directional_bias(
    *,
    field: str,
    old: float,
    value: float,
    raw_delta: float,
    max_delta: float,
    therapist_utterance_type: str,
) -> float:
    if therapist_utterance_type == "logistics":
        if field in {"distress", "emotional_arousal", "safety_risk"} and value > old:
            value = old + min(value - old, max_delta * 0.25)
        return value

    if therapist_utterance_type == "factual_question":
        if field in {"distress", "emotional_arousal", "avoidance_tendency"} and value > old:
            value = old + min(value - old, max_delta * 0.60)
        if field == "therapeutic_alliance":
            value = old + _clip(value - old, -max_delta * 0.50, max_delta * 0.50)
        return value

    if therapist_utterance_type == "reflective_statement":
        if raw_delta >= -0.01:
            if field in {"trust", "perceived_safety", "therapeutic_alliance"}:
                value += min(0.01, max_delta * 0.35)
            elif field == "defensiveness":
                value -= min(0.01, max_delta * 0.35)
        return value

    if therapist_utterance_type == "challenge_or_confrontation":
        if field in {"defensiveness", "emotional_arousal", "distress"}:
            value += min(0.012, max_delta * 0.30)
        elif field in {"trust", "perceived_safety"}:
            value -= min(0.008, max_delta * 0.25)
        elif field == "therapeutic_alliance":
            value = old + _clip(value - old, -max_delta * 0.70, max_delta * 0.70)
        return value

    if therapist_utterance_type == "invalidation":
        if field in {"defensiveness", "distress", "avoidance_tendency"}:
            value += min(0.015, max_delta * 0.35)
        elif field in {"trust", "perceived_safety", "therapeutic_alliance"}:
            value -= min(0.012, max_delta * 0.35)
        return value

    return value


def _contains_avoidance_phrase(text: str) -> bool:
    phrases = [
        "不知道",
        "说不上来",
        "不想说",
        "算了",
        "没什么",
        "i don't know",
        "dont know",
        "not sure",
        "don't want to talk",
        "nothing",
    ]
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _model_dump_json_safe(value: LatentPsychologicalState) -> dict[str, Any]:
    return value.model_dump(mode="json")


def bounded_update_value(
    old: float,
    proposed: float,
    field: str,
    *,
    therapist_utterance_type: str | None = None,
    turn_index: int = 0,
) -> float:
    normalized_type = _normalize_utterance_type(therapist_utterance_type)
    delta = proposed - old
    max_delta = _base_max_delta(field) * TYPE_DELTA_SCALE.get(normalized_type, 0.5)
    clipped_delta = max(-max_delta, min(max_delta, delta))
    bounded = _clip(old + clipped_delta, 0.0, 1.0)
    if field in SLOW_FIELDS:
        bounded = old * 0.75 + bounded * 0.25
    bounded = _apply_directional_bias(
        field=field,
        old=old,
        value=bounded,
        raw_delta=delta,
        max_delta=max_delta,
        therapist_utterance_type=normalized_type,
    )
    bounded = _clip(old + _clip(bounded - old, -max_delta, max_delta), 0.0, 1.0)
    return _apply_soft_bounds(field, bounded)


def adjust_avoidance_by_response_grounding(
    *,
    state_values: dict[str, float],
    previous_state: LatentPsychologicalState,
    client_response: ClientResponse | None,
) -> dict[str, float]:
    if client_response is None or "avoidance_tendency" not in state_values:
        return state_values

    utterance = getattr(client_response, "utterance", "") or ""
    if len(utterance.strip()) <= 15 or _contains_avoidance_phrase(utterance):
        return state_values

    old = float(getattr(previous_state, "avoidance_tendency"))
    new = float(state_values["avoidance_tendency"])
    if new > old:
        adjusted = old + (new - old) * 0.5
        state_values = dict(state_values)
        state_values["avoidance_tendency"] = _clip(adjusted, 0.0, 1.0)
    return state_values


def postprocess_latent_state(
    *,
    previous_state: LatentPsychologicalState,
    raw_state: LatentPsychologicalState,
    therapist_utterance_type: str | None,
    client_response: ClientResponse | None = None,
    turn_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> LatentPsychologicalState:
    del turn_index
    normalized_type = _normalize_utterance_type(therapist_utterance_type)
    type_delta_scale = TYPE_DELTA_SCALE.get(normalized_type, 0.5)
    bounded_payload: dict[str, Any] = raw_state.model_dump(mode="python")
    previous_payload = previous_state.model_dump(mode="python")
    raw_payload = raw_state.model_dump(mode="python")
    numeric_values: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}

    for field in NUMERIC_STATE_FIELDS:
        if field not in previous_payload or field not in raw_payload:
            continue

        old = float(previous_payload[field])
        raw = float(raw_payload[field])
        raw_delta = raw - old
        base_max_delta = _base_max_delta(field)
        applied_max_delta = base_max_delta * type_delta_scale
        clipped_delta = _clip(raw_delta, -applied_max_delta, applied_max_delta)
        bounded = _clip(old + clipped_delta, 0.0, 1.0)

        if field in SLOW_FIELDS:
            bounded = old * 0.75 + bounded * 0.25

        bounded = _apply_directional_bias(
            field=field,
            old=old,
            value=bounded,
            raw_delta=raw_delta,
            max_delta=applied_max_delta,
            therapist_utterance_type=normalized_type,
        )
        bounded = _clip(old + _clip(bounded - old, -applied_max_delta, applied_max_delta), 0.0, 1.0)
        bounded = _apply_soft_bounds(field, bounded)

        bounded_payload[field] = bounded
        numeric_values[field] = bounded
        diagnostics[field] = {
            "old": old,
            "raw": raw,
            "bounded": bounded,
            "raw_delta": raw_delta,
            "bounded_delta": bounded - old,
            "max_delta": applied_max_delta,
            "delta_scale": type_delta_scale,
            "bounds": SOFT_BOUNDS.get(field),
        }

    adjusted_values = adjust_avoidance_by_response_grounding(
        state_values=numeric_values,
        previous_state=previous_state,
        client_response=client_response,
    )
    if adjusted_values is not numeric_values:
        for field, bounded in adjusted_values.items():
            bounded_payload[field] = bounded
            if field in diagnostics:
                old = float(previous_payload[field])
                diagnostics[field]["bounded"] = bounded
                diagnostics[field]["bounded_delta"] = bounded - old
                diagnostics[field]["response_grounding_adjusted"] = True

    bounded_state = LatentPsychologicalState.model_validate(bounded_payload)

    if metadata is not None:
        metadata["state_postprocess"] = {
            "therapist_utterance_type": normalized_type,
            "raw_state": _model_dump_json_safe(raw_state),
            "previous_state": _model_dump_json_safe(previous_state),
            "bounded_state": _model_dump_json_safe(bounded_state),
            "field_updates": diagnostics,
        }

    return bounded_state


def apply_state_update_bounds(
    *,
    previous: LatentPsychologicalState,
    raw: LatentPsychologicalState,
    therapist_utterance_type: str | None,
    session_id: str | int | None,
    turn_index: int,
) -> LatentPsychologicalState:
    del session_id
    return postprocess_latent_state(
        previous_state=previous,
        raw_state=raw,
        therapist_utterance_type=therapist_utterance_type,
        turn_index=turn_index,
    )
