from __future__ import annotations


from collections.abc import Sequence


from pandora.schemas import (
    DialogueTurnRecord,
    LatentPsychologicalState,
    MemoryActivationResult,
    MemoryDisclosureState,
    MemoryItem,
    SupervisionSignal,
)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def update_client_state(
    *,
    previous_state: LatentPsychologicalState,
    latest_therapist_utterance: str,
    recent_dialogue_history: Sequence[DialogueTurnRecord],
    supervision_signal: SupervisionSignal,
    activated_memories: Sequence[MemoryItem],
    memory_activation: MemoryActivationResult,
    memory_disclosure_states: Sequence[MemoryDisclosureState],
) -> LatentPsychologicalState:
    del recent_dialogue_history, latest_therapist_utterance
    evaluation = supervision_signal.therapist_technique_evaluation
    opportunities = supervision_signal.technique_opportunities
    emotion_reflection = evaluation.emotion_reflection.score
    questioning_quality = evaluation.questioning.score
    emotion_reflection_needed = opportunities.listening_opportunities.emotion_reflection.yes
    clarification_needed = opportunities.listening_opportunities.clarification.yes
    paraphrase_needed = opportunities.listening_opportunities.paraphrase.yes
    summary_needed = opportunities.listening_opportunities.summary.yes
    questioning_needed = opportunities.influence_opportunities.questioning.yes
    explanation_needed = opportunities.influence_opportunities.explanation.yes

    trust = previous_state.trust
    defensiveness = previous_state.defensiveness
    arousal = previous_state.emotional_arousal
    disclosure = previous_state.disclosure_readiness
    avoidance = previous_state.avoidance_tendency
    safety = previous_state.perceived_safety
    awareness = previous_state.self_awareness_level
    insight = previous_state.insight_level
    mentalization = previous_state.mentalization_level
    core_belief_activation = previous_state.core_belief_activation
    clarity = previous_state.problem_clarity
    distress = previous_state.distress
    relational_ready = previous_state.perceived_safety >= 0.50 and previous_state.trust >= 0.45

    emotion_reflection_matched = emotion_reflection_needed and emotion_reflection == 1
    emotion_reflection_missed = emotion_reflection_needed and (
        not evaluation.emotion_reflection.used or emotion_reflection == -1
    )
    listening_matched = any(
        (
            clarification_needed and evaluation.clarification.score == 1,
            paraphrase_needed and evaluation.paraphrase.score == 1,
            summary_needed and evaluation.summary.score == 1,
        )
    )
    non_emotion_listening_needed = clarification_needed or paraphrase_needed or summary_needed
    questioning_matched = questioning_needed and questioning_quality == 1
    explanation_matched = explanation_needed and evaluation.explanation.score == 1
    premature_explanation = (
        emotion_reflection_needed
        and not emotion_reflection_matched
        and evaluation.explanation.score == 1
        and not relational_ready
    )
    low_safety_confrontation = evaluation.confrontation.score == 1 and not relational_ready

    if emotion_reflection_missed:
        trust -= 0.12
        defensiveness += 0.15
        disclosure -= 0.12
        avoidance += 0.12
        safety -= 0.14
        arousal += 0.07
        distress += 0.04
        insight -= 0.04
        mentalization -= 0.04
    if emotion_reflection_matched:
        trust += 0.10
        defensiveness -= 0.08
        disclosure += 0.10
        safety += 0.12
        arousal -= 0.05
        mentalization += 0.08
    if listening_matched:
        safety += 0.08
        disclosure += 0.06
        clarity += 0.06
        mentalization += 0.06
    if explanation_matched and relational_ready:
        awareness += 0.12
        clarity += 0.12
        distress -= 0.03
        insight += 0.12
        mentalization += 0.08
    if evaluation.confrontation.score == 1 and opportunities.influence_opportunities.confrontation.yes and relational_ready:
        awareness += 0.06
        clarity += 0.06
        insight += 0.06
    if questioning_matched:
        awareness += 0.05
        clarity += 0.05
    if emotion_reflection_needed and questioning_quality == -1:
        trust -= 0.04
        avoidance += 0.05
        disclosure -= 0.04
    if non_emotion_listening_needed and not listening_matched and evaluation.questioning.score == 1:
        avoidance += 0.05
        clarity -= 0.03
    if premature_explanation:
        trust -= 0.05
        safety -= 0.05
        defensiveness += 0.10
        arousal += 0.08
        avoidance += 0.04
        insight -= 0.04
    if low_safety_confrontation:
        trust -= 0.06
        safety -= 0.06
        defensiveness += 0.14
        arousal += 0.08
        avoidance += 0.06
        insight -= 0.03

    # information, immediacy, and self_disclosure are intentionally not mapped
    # to direct latent-state deltas here. Plain information should not be
    # treated as effective analysis or insight-building by itself. Immediacy is
    # reserved for future relationship rupture/repair modeling. Self-disclosure
    # depends heavily on timing and boundaries, so adding a mechanical positive
    # or negative update would be misleading.

    if activated_memories:
        highest_disclosure = max(memory.disclosure_level for memory in activated_memories)
        arousal += 0.02 * highest_disclosure
        distress += 0.015 * highest_disclosure
        core_belief_activation += 0.08 * highest_disclosure
        if memory_activation.disclosure_permission_level >= highest_disclosure:
            disclosure += 0.05
        if highest_disclosure >= 2 and safety >= 0.45:
            awareness += 0.05
            insight += 0.05
        elif highest_disclosure >= 2:
            avoidance += 0.05

    if memory_disclosure_states:
        max_emotional_activation = max(item.emotional_activation_strength for item in memory_disclosure_states)
        max_accessibility = max(item.cognitive_accessibility for item in memory_disclosure_states)
        arousal += 0.08 * max_emotional_activation
        core_belief_activation += 0.10 * max_emotional_activation
        if any(item.narratability == "felt_but_unnamed" for item in memory_disclosure_states):
            avoidance += 0.05
            distress += 0.03
        if any(item.narratability in {"partially_narrated", "clearly_narrated"} for item in memory_disclosure_states):
            awareness += 0.06
            insight += 0.08
            mentalization += 0.06
            clarity += 0.05
        if any(item.disclosure_mode == "core_belief_statement" for item in memory_disclosure_states):
            insight += 0.06
            core_belief_activation += 0.08
        elif max_accessibility < 0.4 and max_emotional_activation >= 0.55:
            defensiveness += 0.04

    dominant_emotions = list(previous_state.dominant_emotions)
    if emotion_reflection_missed:
        dominant_emotions = ["guarded", "hurt", *dominant_emotions]
    elif emotion_reflection_matched:
        dominant_emotions = ["understood", *dominant_emotions]
    elif activated_memories:
        dominant_emotions = ["activated", *dominant_emotions]
    if any(item.narratability == "felt_but_unnamed" for item in memory_disclosure_states):
        dominant_emotions = ["confused", *dominant_emotions]

    presenting_concerns = list(previous_state.presenting_concerns)
    interpersonal_patterns = list(previous_state.interpersonal_patterns)
    current_needs = list(previous_state.current_needs)
    core_beliefs = list(previous_state.core_beliefs)

    for memory in activated_memories:
        presenting_concerns.extend(memory.emotion_tags[:1])
        interpersonal_patterns.extend(memory.mechanism_tags[:2])
        current_needs.extend(memory.need_tags[:2])
        core_beliefs.extend(memory.core_belief_tags[:2])

    if emotion_reflection_missed:
        current_needs.append("to feel emotionally understood before going deeper")
    elif emotion_reflection_matched:
        current_needs.append("to keep exploring while feeling understood")

    dominant_emotions = _unique(dominant_emotions)[:4]
    current_needs = _unique(current_needs)[:4]
    core_beliefs = _unique(core_beliefs)[:4]
    presenting_concerns = _unique(presenting_concerns)[:4]
    interpersonal_patterns = _unique(interpersonal_patterns)[:4]

    active_core_belief = previous_state.active_core_belief
    if not active_core_belief and core_beliefs:
        active_core_belief = core_beliefs[0]
    for memory in activated_memories:
        if memory.core_belief_tags:
            active_core_belief = memory.core_belief_tags[0]
            break

    surface_focus = presenting_concerns[0] if presenting_concerns else "this"
    if awareness < 0.4:
        current_self_explanation = f"Maybe I am just handling {surface_focus} badly."
    elif awareness < 0.7 and active_core_belief:
        current_self_explanation = (
            f"It feels like more than {surface_focus}, but I cannot fully explain why it hits me this hard."
        )
    elif active_core_belief:
        current_self_explanation = (
            f"Part of me thinks {surface_focus} stirs up the fear that I am {active_core_belief}."
        )
    else:
        current_self_explanation = previous_state.current_self_explanation

    return LatentPsychologicalState(
        distress=_clamp(distress),
        trust=_clamp(trust),
        defensiveness=_clamp(defensiveness),
        emotional_arousal=_clamp(arousal),
        disclosure_readiness=_clamp(disclosure),
        avoidance_tendency=_clamp(avoidance),
        perceived_safety=_clamp(safety),
        self_awareness_level=_clamp(awareness),
        insight_level=_clamp(insight),
        mentalization_level=_clamp(mentalization),
        core_belief_activation=_clamp(core_belief_activation),
        problem_clarity=_clamp(clarity),
        dominant_emotion=dominant_emotions[0] if dominant_emotions else previous_state.dominant_emotion,
        dominant_need=current_needs[0] if current_needs else previous_state.dominant_need,
        active_core_belief=active_core_belief,
        current_self_explanation=current_self_explanation,
        dominant_emotions=dominant_emotions,
        presenting_concerns=presenting_concerns,
        interpersonal_patterns=interpersonal_patterns,
        current_needs=current_needs,
        core_beliefs=core_beliefs,
    )
