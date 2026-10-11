from __future__ import annotations


from enum import Enum


from typing import Annotated, Any, Literal


from pydantic import BaseModel, ConfigDict, Field, StringConstraints


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


TechniqueLabel = Literal[
    "clarification",
    "paraphrase",
    "emotion_reflection",
    "summary",
    "questioning",
    "explanation",
    "information",
    "immediacy",
    "self_disclosure",
    "confrontation",
]


CounselorTechnique = Literal["empathy", "listening_summary", "questioning", "analysis"]


TechniqueScore = Literal[-1, 0, 1]


DisclosureLevel = Literal[0, 1, 2, 3]


AttributionTarget = Literal[
    "self_trait",
    "self_behavior",
    "other_person",
    "environment",
    "relationship",
    "unclear",
    "core_belief",
]


ClientStance = Literal[
    "open",
    "hesitant",
    "avoidant",
    "defensive",
    "confused",
    "intellectualized",
    "emotionally_overwhelmed",
]


SpeechStyle = Literal[
    "direct",
    "vague",
    "hesitant",
    "contradictory",
    "intellectualized",
    "emotionally_flooded",
    "brief_withdrawn",
]


Narratability = Literal[
    "inaccessible",
    "felt_but_unnamed",
    "hinted",
    "partially_narrated",
    "clearly_narrated",
]


DisclosureMode = Literal[
    "none",
    "somatic_hint",
    "emotion_hint",
    "vague_reference",
    "partial_event",
    "explicit_event",
    "core_belief_statement",
]


CognitiveDistortion = Literal[
    "personalization",
    "overgeneralization",
    "mind_reading",
    "catastrophizing",
    "emotional_reasoning",
    "should_statement",
    "all_or_nothing",
    "none",
]


CounselorErrorType = Literal[
    "missing_empathy",
    "excessive_questioning",
    "premature_analysis",
    "premature_advice",
    "topic_shift",
    "premature_confrontation",
    "over_self_disclosure",
]


DialogueMode = Literal["offline", "closed_loop"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MemoryKind(str, Enum):
    SURFACE = "surface"
    MECHANISM = "mechanism"
    LATENT_CLINICAL = "latent_clinical"


class AblationMode(str, Enum):
    NO_MEMORY = "no_memory"
    SURFACE_MEMORY = "surface_memory"
    MECHANISM_MEMORY = "mechanism_memory"
    FULL_MEMORY = "full_memory"


class ClientGenerationMode(str, Enum):
    DIRECT = "direct"
    STATE_ONLY = "state_only"
    EXPRESSION_PLANNED = "expression_planned"
    FULL_PANDORA = "full_pandora"


class ClientProfile(StrictModel):
    client_id: NonEmptyStr = "client-1"
    background_summary: str | None = None
    presenting_themes: list[NonEmptyStr] = Field(default_factory=list)
    therapy_goals: list[NonEmptyStr] = Field(default_factory=list)


class LatentPsychologicalState(StrictModel):
    distress: float = Field(ge=0.0, le=1.0)
    trust: float = Field(ge=0.0, le=1.0)
    defensiveness: float = Field(ge=0.0, le=1.0)
    emotional_arousal: float = Field(ge=0.0, le=1.0)
    disclosure_readiness: float = Field(ge=0.0, le=1.0)
    avoidance_tendency: float = Field(ge=0.0, le=1.0)
    perceived_safety: float = Field(ge=0.0, le=1.0)
    self_awareness_level: float = Field(ge=0.0, le=1.0)
    insight_level: float = Field(default=0.0, ge=0.0, le=1.0)
    mentalization_level: float = Field(default=0.0, ge=0.0, le=1.0)
    core_belief_activation: float = Field(default=0.0, ge=0.0, le=1.0)
    problem_clarity: float = Field(ge=0.0, le=1.0)
    dominant_emotion: NonEmptyStr = "uncertain"
    dominant_need: NonEmptyStr = "to feel understood"
    active_core_belief: str | None = None
    current_self_explanation: NonEmptyStr = "I am not sure why this affects me so much."
    dominant_emotions: list[NonEmptyStr] = Field(default_factory=list)
    presenting_concerns: list[NonEmptyStr] = Field(default_factory=list)
    interpersonal_patterns: list[NonEmptyStr] = Field(default_factory=list)
    current_needs: list[NonEmptyStr] = Field(default_factory=list)
    core_beliefs: list[NonEmptyStr] = Field(default_factory=list)


class MemoryItem(StrictModel):
    memory_id: NonEmptyStr
    kind: MemoryKind
    content: NonEmptyStr
    accessibility: MemoryKind | None = None
    type: str | None = None
    memory_type: str | None = None
    session_id: int | None = Field(default=None, ge=1)
    source_session_id: int | None = Field(default=None, ge=1)
    visibility_start_session: int | None = Field(default=None, ge=1)
    available_from_session: int | None = Field(default=None, ge=1)
    visibility_start_turn: int | None = Field(default=None, ge=0)
    source: str | None = None
    source_path: str | None = None
    client_visibility: str | None = None
    expression_permission: str | None = None
    clinical_relevance: str | None = None
    emotional_tags: list[NonEmptyStr] = Field(default_factory=list)
    interpersonal_tags: list[NonEmptyStr] = Field(default_factory=list)
    verbatim_quote: bool = False
    surface_cues: list[NonEmptyStr] = Field(default_factory=list)
    emotion_tags: list[NonEmptyStr] = Field(default_factory=list)
    need_tags: list[NonEmptyStr] = Field(default_factory=list)
    core_belief_tags: list[NonEmptyStr] = Field(default_factory=list)
    mechanism_tags: list[NonEmptyStr] = Field(default_factory=list)
    disclosure_level: DisclosureLevel
    activation_threshold: float = Field(ge=0.0, le=1.0)
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    source_turn: int | None = Field(default=None, ge=0)
    evidence: list[NonEmptyStr] = Field(default_factory=list)


ClientMemory = MemoryItem


class TechniqueOpportunityDecision(StrictModel):
    yes: bool
    reason: NonEmptyStr


class ListeningTechniqueOpportunities(StrictModel):
    clarification: TechniqueOpportunityDecision
    paraphrase: TechniqueOpportunityDecision
    emotion_reflection: TechniqueOpportunityDecision
    summary: TechniqueOpportunityDecision


class InfluenceTechniqueOpportunities(StrictModel):
    questioning: TechniqueOpportunityDecision
    explanation: TechniqueOpportunityDecision
    information: TechniqueOpportunityDecision
    immediacy: TechniqueOpportunityDecision
    self_disclosure: TechniqueOpportunityDecision
    confrontation: TechniqueOpportunityDecision


class TechniqueOpportunitySignal(StrictModel):
    listening_technique_needed: TechniqueOpportunityDecision
    influence_technique_needed: TechniqueOpportunityDecision
    listening_opportunities: ListeningTechniqueOpportunities
    influence_opportunities: InfluenceTechniqueOpportunities


class TechniqueUseScore(StrictModel):
    used: bool
    score: TechniqueScore
    reason: NonEmptyStr
    evidence: NonEmptyStr


class RealTherapistTechniqueEvaluation(StrictModel):
    clarification: TechniqueUseScore
    paraphrase: TechniqueUseScore
    emotion_reflection: TechniqueUseScore
    summary: TechniqueUseScore
    questioning: TechniqueUseScore
    explanation: TechniqueUseScore
    information: TechniqueUseScore
    immediacy: TechniqueUseScore
    self_disclosure: TechniqueUseScore
    confrontation: TechniqueUseScore
    primary_techniques: list[TechniqueLabel] = Field(default_factory=list)
    overall_reason: NonEmptyStr


class SupervisionSignal(StrictModel):
    technique_opportunities: TechniqueOpportunitySignal
    therapist_technique_evaluation: RealTherapistTechniqueEvaluation


class ObservableClientState(StrictModel):
    expressed_emotion: NonEmptyStr
    apparent_attribution: NonEmptyStr
    disclosure_depth: DisclosureLevel
    resistance_level: float = Field(ge=0.0, le=1.0)
    topic_focus: NonEmptyStr
    uncertainty_level: float = Field(ge=0.0, le=1.0)


class CounselorSkillProfile(StrictModel):
    empathy_skill: float = Field(ge=0.0, le=1.0)
    listening_skill: float = Field(ge=0.0, le=1.0)
    questioning_skill: float = Field(ge=0.0, le=1.0)
    analysis_skill: float = Field(ge=0.0, le=1.0)
    timing_awareness: float = Field(ge=0.0, le=1.0)
    risk_awareness: float = Field(ge=0.0, le=1.0)
    self_disclosure_control: float = Field(ge=0.0, le=1.0)
    tendency_to_advise: float = Field(ge=0.0, le=1.0)
    tendency_to_question: float = Field(ge=0.0, le=1.0)
    tendency_to_analyze: float = Field(ge=0.0, le=1.0)


class CounselorUtterance(StrictModel):
    utterance: NonEmptyStr
    intended_technique: CounselorTechnique
    counselor_intent: NonEmptyStr
    error_type: CounselorErrorType | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class TrainingFeedback(StrictModel):
    triggered: bool = False
    main_issue: str | None = None
    feedback: str | None = None
    suggested_revision: str | None = None


class MemoryActivationScore(StrictModel):
    memory_id: NonEmptyStr
    surface_cue_match: float = Field(ge=0.0, le=1.0)
    emotion_match: float = Field(ge=0.0, le=1.0)
    need_match: float = Field(ge=0.0, le=1.0)
    core_belief_match: float = Field(ge=0.0, le=1.0)
    mechanism_match: float = Field(ge=0.0, le=1.0)
    supervision_safety_adjustment: float = Field(ge=-1.0, le=1.0)
    total_activation_score: float = Field(ge=0.0, le=1.0)
    disclosure_allowed: bool
    blocked_reason: str | None = None


class MemoryActivationResult(StrictModel):
    activated_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    disclosure_permission_level: DisclosureLevel
    activation_reason: NonEmptyStr
    blocked_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    blocked_reason: NonEmptyStr
    memory_scores: list[MemoryActivationScore] = Field(default_factory=list)


class MemoryDisclosureState(StrictModel):
    memory_id: NonEmptyStr
    activated: bool
    emotional_activation_strength: float = Field(ge=0.0, le=1.0)
    cognitive_accessibility: float = Field(ge=0.0, le=1.0)
    narratability: Narratability
    disclosure_allowed: bool
    disclosure_mode: DisclosureMode
    reason: NonEmptyStr


class SubjectiveAppraisal(StrictModel):
    perceived_event: NonEmptyStr
    perceived_threat: NonEmptyStr
    dominant_emotion: NonEmptyStr
    secondary_emotion: str | None = None
    perceived_self_implication: NonEmptyStr
    perceived_interpersonal_implication: NonEmptyStr
    appraisal_confidence: float = Field(ge=0.0, le=1.0)


class AttributionFrame(StrictModel):
    attribution_target: AttributionTarget
    attribution_statement: NonEmptyStr
    is_misattribution: bool
    cognitive_distortion: CognitiveDistortion | None = None
    hidden_underlying_belief: str | None = None
    awareness_gap: NonEmptyStr


class DisclosurePlan(StrictModel):
    disclosure_depth: DisclosureLevel
    disclose_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    hint_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    avoid_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    allowed_content: NonEmptyStr
    forbidden_content: NonEmptyStr
    expected_client_stance: ClientStance


class LinguisticRealizationPlan(StrictModel):
    speech_style: SpeechStyle
    pronoun_focus: Literal["I", "others", "situation", "mixed"]
    certainty_level: Literal["low", "medium", "high"]
    emotional_explicitness: Literal["low", "medium", "high"]
    hedging_level: Literal["low", "medium", "high"] = "medium"
    target_length_chars: int | None = Field(default=None, ge=1)
    target_length_min_chars: int | None = Field(default=None, ge=1)
    target_length_max_chars: int | None = Field(default=None, ge=1)
    hard_max_length_chars: int | None = Field(default=None, ge=1)
    target_sentence_count: int | None = Field(default=None, ge=1)
    preferred_markers: list[NonEmptyStr] = Field(default_factory=list)
    habitual_style_repetition_allowed: bool = True
    avoid_semantic_repetition: bool = True
    metaphor_or_body_expression: str | None = None
    utterance_constraints: list[NonEmptyStr] = Field(default_factory=list)


class ClientExpressionPlan(StrictModel):
    subjective_appraisal: SubjectiveAppraisal
    attribution_frame: AttributionFrame
    disclosure_plan: DisclosurePlan
    linguistic_realization_plan: LinguisticRealizationPlan


class ClientResponse(StrictModel):
    utterance: NonEmptyStr
    expressed_emotion: NonEmptyStr
    apparent_attribution: NonEmptyStr
    disclosure_depth: DisclosureLevel
    hidden_unspoken_content: str = ""
    used_memory_ids: list[NonEmptyStr] = Field(default_factory=list)
    affect: list[NonEmptyStr] = Field(default_factory=list)
    self_disclosure: float = Field(ge=0.0, le=1.0)
    referenced_memory_ids: list[NonEmptyStr] = Field(default_factory=list)


class EvaluatorOutput(StrictModel):
    coherence: float = Field(ge=0.0, le=1.0)
    emotional_congruence: float = Field(ge=0.0, le=1.0)
    therapeutic_alliance: float = Field(ge=0.0, le=1.0)
    realism: float = Field(ge=0.0, le=1.0)
    safety_risk: float = Field(ge=0.0, le=1.0)
    notes: NonEmptyStr


class EncodedMemoryBatch(StrictModel):
    memories: list[MemoryItem] = Field(default_factory=list)


class DialogueTurnRecord(StrictModel):
    turn_index: int = Field(ge=0)
    dialogue_mode: DialogueMode = "offline"
    ablation_mode: AblationMode
    generation_mode: ClientGenerationMode
    counselor_profile_name: NonEmptyStr | None = None
    counselor_skill_profile: CounselorSkillProfile | None = None
    forced_error_type: CounselorErrorType | None = None
    observable_client_state: ObservableClientState | None = None
    counselor_utterance: CounselorUtterance | None = None
    therapist_utterance: NonEmptyStr
    supervision_signal: SupervisionSignal
    memory_activation: MemoryActivationResult
    activated_memories: list[MemoryItem] = Field(default_factory=list)
    memory_disclosure_states: list[MemoryDisclosureState] = Field(default_factory=list)
    latent_state_before: LatentPsychologicalState
    latent_state_after: LatentPsychologicalState
    subjective_appraisal: SubjectiveAppraisal
    attribution_frame: AttributionFrame
    disclosure_plan: DisclosurePlan
    linguistic_realization_plan: LinguisticRealizationPlan
    client_response: ClientResponse
    training_feedback: TrainingFeedback = Field(default_factory=TrainingFeedback)
    evaluator_scores: EvaluatorOutput
    encoded_memories: list[MemoryItem] = Field(default_factory=list)
    response_diagnostics: dict[str, Any] = Field(default_factory=dict)
    therapist_anchor: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DialogueSession(StrictModel):
    dialogue_mode: DialogueMode = "offline"
    ablation_mode: AblationMode
    generation_mode: ClientGenerationMode
    output_path: NonEmptyStr
    turns: list[DialogueTurnRecord] = Field(default_factory=list)
    final_latent_state: LatentPsychologicalState
    final_memory_bank: list[MemoryItem] = Field(default_factory=list)
