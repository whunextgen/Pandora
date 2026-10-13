# From source dialogue to simulation packages

The preprocessing implementation is included alongside the modeling core. It does not distribute source dialogues, participant records, case-specific profiles, generated skills from real cases, local configuration, or credentials.

## Pipeline

1. **Normalize the source dialogue.** `src/pandora_data/normalize.py` provides adapters for PsyDial-D4, AnnoMI, ConsistentMI, and Eeyore. The adapters map each source's speaker convention to `client`/`counselor`, preserve source fields, and assign continuous message indices. In particular, the `user` role has different meanings in PsyDial and Eeyore. `schema.py` validates roles, nonempty messages, indices, and provenance.
2. **Materialize case info.** `scripts/materialize_case_folders.py` consumes a normalized, selected case and its reviewed evidence. `build_client_info` uses the existing source profile, selected client quotes, and review rationale; it does not infer missing demographics. `build_dialog` preserves utterances and the source counselor system prompt. `materialize_case` writes `client_info.json`, `dialog.json`, the canonical `original_dialog.json`, selection-evidence cards, ordering evidence, and manifests. Selection and evidence annotations are inputs to this step, not predictions made by the materializer.
3. **Extract client-grounded runtime memories.** `scripts/extract_client_grounded_memories.py` contains the exact extraction prompt, output schemas, retry/validation logic, deduplication, and memory-bank construction. It partitions the dialogue into three contiguous segments of client-completed exchanges. These segments organize extraction; they are not clinical session labels or runtime release gates. The prompt asks for sparse, minimally self-contained autobiographical facts explicitly stated or endorsed by the client, in the source language and first person. Counselor suggestions and unsupported clinical explanations are excluded. Supporting client-turn indices and output language are validated. Runtime cards have `activation_threshold=0.0`; their level records privacy/vulnerability. The order file preserves source ordering for inspection, without controlling the released full-memory policy.
4. **Distill the client package.** `scripts/preprocess_psydial_cases.py:build_client` computes client-only turn-length and habitual-marker statistics. It assembles a neutral profile, case-content profile, language-style description, interaction and disclosure instructions, evidence register, and initial state. Concerns, patterns, and beliefs are populated from the supplied memory levels. Initial numerical values are implementation defaults. These are deterministic templates and observed statistics, not an additional LLM inference step.
5. **Distill the therapist package.** `scripts/distill_case_therapists.py` reads `client_info.json` and `dialog.json`, preserves the counselor's source system prompt, extracts counselor-only statistics and representative utterances, and assembles strategy, language-style, process-progression, and evidence files. Intent categories are keyword-based retrieval labels, not expert annotations. `preprocess_psydial_cases.py` supplies a generic, source-grounded fallback for dialogues without a counselor system prompt.
6. **Organize the outputs.** `preprocess` creates `1case_info/`, `2client_distillation/`, and `2therapist_distillation/` inside each case directory. These packages provide the case content, habitual client expression evidence, and source therapist behavior used by the simulator.

## Run the complete wiring check

After extracting the archive:

```sh
python -m pip install -r requirements.txt
python preprocessing_demo.py
```

The example constructs an invented raw dialogue, normalizes it, attaches synthetic selection evidence, materializes case info, validates synthetic structured memory responses, and builds both distilled packages. Assertions check role-specific statistics, source-prompt preservation, populated client-state fields, and rejection of third-person memory text. A stub replaces the LLM; the example checks pipeline wiring and does not measure extraction quality or reproduce experiment results. Temporary generated files are removed automatically.

## Apply the functions to your own inputs

For a selected normalized case `case` with `selection.evidence.L1/L2/L3` (each containing valid client message indices and exact quotes):

```python
from pathlib import Path
from scripts.materialize_case_folders import materialize_case
from scripts.extract_client_grounded_memories import generate_case
from scripts.preprocess_psydial_cases import preprocess

materialize_case(case, line_number=1, input_path=Path("selected_cases.jsonl"),
                 output_root=Path("cases"))
case_root = Path("cases") / case["case_id"].replace(":", "__")
generate_case(case_dir=case_root, provider=provider,
              provider_model="your-model", overwrite=False)
preprocess(case_root)
```

`provider` is a caller-supplied structured-output adapter exposing `_parse_with_openai(prompt=..., schema=...)` and returning an instance of the requested Pydantic schema. The historical method name is only an interface: the release contains no model-provider configuration or API key. `build_prompt` and `GroundedFactSummary` can also be used directly with another structured-output interface. Credentials remain in the caller's environment. Only local provider loading and the project-specific CLI were excluded from the original extraction script; the retained definitions preserve their source bodies.

Source adapters and case builders preserve supplied profiles, prompts, quotes, provenance, and reviewer fields. They do not anonymize arbitrary inputs. Review and de-identify your own inputs and generated outputs before sharing them. The public artifact contains only code and the invented wiring example.

## Historical metadata

The case materializer retains legacy selection-evidence thresholds and explanatory strings from the source implementation. They describe selection/audit artifacts, not the runtime memory-access policy. The released model makes all currently available memories accessible, as described in the main README. No original data-selection decisions or private annotation files are included.
