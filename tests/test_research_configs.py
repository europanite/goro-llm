import json
from pathlib import Path


def test_minami2024_paper_config_core_values():
    cfg = json.loads(Path("configs/minami2024_reading.json").read_text())
    assert cfg["model_name"] == "cyberagent/calm2-7b-chat"
    assert cfg["lora"]["r"] == 16
    assert cfg["lora"]["alpha"] == 64
    assert cfg["lora"]["target_modules"] == ["q_proj", "k_proj", "embed_tokens", "lm_head"]
    assert cfg["training"]["warmup_ratio"] == 0.2
    assert cfg["training"]["num_train_epochs"] == 1


def test_3090_configs_use_4bit():
    for name in ["gemma2_jpn_sft_3090.json", "gemma2_jpn_dpo_3090.json"]:
        cfg = json.loads((Path("configs") / name).read_text())
        assert cfg["quantization"]["load_in_4bit"] is True
        assert cfg["training"]["per_device_train_batch_size"] == 1
