"""Exercise the exported pipeline using invented dialogue and a stub model response."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.pandora_data.normalize import normalize_psydial
from scripts.materialize_case_folders import materialize_case
from scripts.extract_client_grounded_memories import (
    GroundedFactSummary, PHASES, build_prompt, generate_case, split_dialogue_phases,
)
from scripts.preprocess_psydial_cases import preprocess


class SyntheticProvider:
    """Return invented facts; this checks wiring, not LLM extraction quality."""
    def __init__(self, responses):
        self.responses = iter(responses)

    def _parse_with_openai(self, *, prompt, schema):
        assert "Dialogue:" in prompt.user
        return schema.model_validate(next(self.responses))


def main():
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        raw = root / "synthetic_dialogue.json"
        clients = [
            "我最近每天上班前都会紧张。", "我昨天准备报告时一直睡不着。",
            "我常常反复检查自己写的报告。", "我每次收到反馈都会先觉得自己做错了。",
            "我从小就担心让家人失望。", "我总觉得只有做得很好才值得被认可。",
        ]
        messages = [{"role": "system", "content": "Respond as a supportive counselor; explore one concern at a time."}]
        for utterance in clients:
            messages.extend([{"role": "assistant", "content": "你能说说当时的感受吗？"},
                             {"role": "user", "content": utterance}])
        raw.write_text(json.dumps([{"messages": messages}], ensure_ascii=False), encoding="utf-8")
        case = next(normalize_psydial(raw))
        case["selection"] = {"status": "synthetic_wiring_example", "evidence": {
            layer: [{"message_index": index, "quote": case["messages"][index]["content"]}]
            for layer, index in zip(("L1", "L2", "L3"), (1, 5, 9))
        }}
        materialize_case(case, line_number=1, input_path=raw, output_root=root / "cases")
        case_root = root / "cases" / case["case_id"].replace(":", "__")
        dialog = json.loads((case_root / "dialog.json").read_text(encoding="utf-8"))
        phases = split_dialogue_phases(dialog["dialog"])
        responses = []
        for phase, level in zip(PHASES, (1, 2, 3)):
            turn = next(t for t in phases[phase] if t["speaker"] == "client")
            prompt = build_prompt(dataset_id="synthetic", phase=phase, turns=phases[phase])
            assert turn["utterance"] in prompt.user
            responses.append({"memories": [{"fact_kind": "experience", "disclosure_level": level,
                "content": turn["utterance"], "source_client_turns": [turn["turn_id"]]}]})
        generate_case(case_dir=case_root, provider=SyntheticProvider(responses),
                      provider_model="synthetic-stub", overwrite=False)
        preprocess(case_root)
        info = case_root / "1case_info"
        memories = json.loads((info / "memory_bank.json").read_text(encoding="utf-8"))
        assert len(memories) == 3 and all(m["activation_threshold"] == 0 for m in memories)
        client_stats = json.loads((case_root / "2client_distillation/evidence/behavioral_stats.json").read_text(encoding="utf-8"))
        therapist_stats = json.loads((case_root / "2therapist_distillation/evidence/behavioral_stats.json").read_text(encoding="utf-8"))
        assert client_stats["client_turn_count"] == 6
        assert therapist_stats["therapist_turn_count"] == 6
        assert (case_root / "2therapist_distillation/source_system_prompt.txt").read_text(encoding="utf-8") == messages[0]["content"]
        state = json.loads((info / "client_state.json").read_text(encoding="utf-8"))
        assert state["presenting_concerns"] == [clients[0]]
        assert state["interpersonal_patterns"] == [clients[2]]
        assert state["core_beliefs"] == [clients[4]]
        try:
            GroundedFactSummary.model_validate({"memories": [{"fact_kind": "experience",
                "disclosure_level": 1, "content": "The client is anxious.", "source_client_turns": [1]}]})
        except ValueError:
            pass
        else:
            raise AssertionError("third-person memory must fail validation")
    print("Synthetic preprocessing checks passed: raw dialogue -> case info -> client and therapist packages.")


if __name__ == "__main__":
    main()
