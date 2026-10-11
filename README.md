# PAnDora: psychological state and expression modeling

This artifact contains PAnDora's deterministic modeling core: psychological state updates, subjective appraisal, attribution, content planning, and linguistic expression planning. The functions are extracted from the implementation with their bodies preserved; experiment runners, model-provider integrations, datasets, participant records, local configuration, and repository history are excluded.

## Run a synthetic example

Download `pandora-modeling-anonymous.zip` from this repository and extract it first. The archive contains the complete Python package with its directory structure. The state-transition source files are also provided at the repository root for convenient inspection; run the package from the extracted directory.

```sh
unzip pandora-modeling-anonymous.zip
cd pandora-modeling
python -m pip install -r requirements.txt
python demo.py
```

The example uses invented utterances, runs locally, and requires no API key. It checks contrasting therapist responses, state bounds, unrestricted access to currently available memories, and expression-plan construction. It does not generate LLM responses or rerun the paper's experiments.

## The 12 numerical state dimensions

All dimensions use a normalized [0, 1] scale. The state also carries emotion, need, belief, concern, and self-explanation text fields.

| Dimension | Meaning in the model | Base single-turn cap | Soft range |
|---|---|---:|---|
| `distress` | Subjective distress | .060 | [0, .92] |
| `emotional_arousal` | Emotional activation | .060 | [0, .88] |
| `defensiveness` | Defensive orientation | .050 | [0, .90] |
| `avoidance_tendency` | Tendency to avoid engagement | .050 | [0, .85] |
| `trust` | Trust in the therapist | .035 | [.10, .90] |
| `perceived_safety` | Felt interpersonal safety | .035 | [.05, .85] |
| `disclosure_readiness` | Readiness to communicate personal material | .040 | [0, 1] |
| `self_awareness_level` | Awareness of one's own experience | .040 | [0, 1] |
| `insight_level` | Understanding of recurring patterns | .030 | [0, 1] |
| `mentalization_level` | Capacity to interpret mental states | .030 | [0, 1] |
| `core_belief_activation` | Current activation of a core belief | .050 | [0, 1] |
| `problem_clarity` | Clarity about the presenting difficulty | .040 | [0, 1] |

These values are implementation parameters, rather than clinically estimated scale scores or cutoffs. `therapeutic_alliance` and `safety_risk` remain compatibility keys in the postprocessing constants; they are not dimensions of `LatentPsychologicalState` and are not updated by its numerical-field loop.

## How the state updates

The active update runs **before** the next client response and consists of two stages.

1. **Propose a state.** `pandora/client/state_update.py:update_client_state` applies additive rules to the previous state using therapist-technique opportunities and performance, available memories, and memory descriptors. For example, missing an indicated emotion reflection proposes trust -.12, defensiveness +.15, readiness -.12, and safety -.14. A matched reflection proposes trust +.10, defensiveness -.08, readiness +.10, and safety +.12. Matched explanation increases awareness and insight only when previous safety >= .50 and trust >= .45. Activated memory depth and tags additionally affect affective, cognitive, and text fields. The complete rule list is in the function; the proposed numerical state is clipped to [0, 1].
2. **Bound and smooth the proposal.** `pandora/client_simulation/state_layer.py:postprocess_latent_state` clips each proposed change, smooths trust and safety, applies therapist-type directional adjustments, clips the change again, and applies the soft ranges above.

For dimension i, previous value x, raw proposal r, and effective cap m:

```text
m = base_cap[i] * therapist_type_scale
y = clip(x + clip(r - x, -m, m), 0, 1)
if i is trust or perceived_safety:
    y = .75 * x + .25 * y
y = directional_adjustment(i, x, y, r - x, m, therapist_type)
y = clip(x + clip(y - x, -m, m), 0, 1)
next_value = clip(y, soft_lower[i], soft_upper[i])
```

The therapist-type scales are .10 for logistics, .40 for factual questions, .50 for reflective statements, .80 for challenges/confrontations, and 1.00 for invalidation. `therapist_input.py` contains the lexical classifier and the exact mapping used by the adapter. Emotion questions, reflections, and normalization map to the reflective category; unlisted categories default to factual questions. Technique assessment is a separate heuristic signal (`supervisor/supervisor.py`), which can also be supplied through its validated schemas.

Directional adjustments attenuate affective increases for logistics and factual questions; supportive reflective turns can increase trust/safety and reduce defensiveness; challenges and invalidation introduce corresponding defensive/affective biases. These adjustments occur after slow-field smoothing. Consequently, slow-field changes are not simply one quarter of the cap. If an initial value is outside a soft range, the final soft-range projection can also exceed the nominal single-turn cap. The optional client-response avoidance adjustment is available in the helper, but is not used in this pre-response update path (`client_response=None`).

## Memory access and expression planning

There is **no psychological memory-access or gradual-disclosure gate** in this version. `full_memory_oracle_policy` activates every currently available memory with accessibility and activation strength 1, sets disclosure permission to 3, and blocks none. Temporal visibility filtering only excludes events from future sessions/turns; it does not condition access on trust, readiness, or emotion. Legacy threshold fields in the shared schemas do not constrain this policy. The safety/trust condition above concerns the effect of explanation on state, rather than memory availability.

After the bounded update, `client/expression_planning.py` constructs subjective appraisal, attribution, and a content plan from the updated state and available memories. `client_simulation/response_act_layer.py` specifies the response act; `linguistic_layer.py` and `expression_planning_module.py` determine uncertainty, hedging, emotional explicitness, length, and other expression constraints using state and optional individual language statistics. This artifact exposes these plans; model calls and utterance realization remain outside its scope.

## Files

- `pandora/schemas.py`: validated structures, including the psychological state.
- `pandora/client/state_update.py`: raw state transition rules.
- `pandora/client_simulation/state_layer.py`: caps, smoothing, directional adjustments, and bounds.
- `pandora/supervisor/supervisor.py`: heuristic technique signals and structured-signal validation.
- `pandora/client/memory_activation.py`, `client_simulation/memory_layer.py`: unrestricted current-memory access and temporal visibility.
- `pandora/client/expression_planning.py`: appraisal, attribution, and content planning.
- `pandora/client_simulation/{response_act_layer,linguistic_layer,expression_planning_module}.py`: response acts and situated expression planning.

Only modeling definitions and a synthetic local example are distributed. No original case content, identifying project metadata, credentials, or Git history is included.
