#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

from goro_llm.training_data import pun_sft_messages, reading_messages


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA/QLoRA SFT runner for the research configs")
    p.add_argument("config", type=Path)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("Install training extras: pip install -e '.[train]'") from exc

    quant = cfg.get("quantization", {})
    quant_cfg = None
    if quant.get("load_in_4bit"):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, quant.get("compute_dtype", "bfloat16")),
            bnb_4bit_use_double_quant=quant.get("double_quant", True),
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], device_map="auto", torch_dtype="auto",
        quantization_config=quant_cfg,
    )
    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0),
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM",
        modules_to_save=lora.get("modules_to_save"),
    )
    ds = load_dataset("json", data_files=cfg["data_files"])
    task = cfg["task"]
    seed = int(cfg.get("seed", 42))

    def render(row):
        if task == "reading_bidirectional":
            # Matches the paper's equal-probability choice of the two reading tasks.
            digest = hashlib.sha256(f"{seed}:{row.get('text', '')}".encode("utf-8")).digest()
            reverse = (digest[0] / 255.0) < 0.5
            messages = reading_messages(row, reverse=reverse)
        elif task == "pun_sft":
            messages = pun_sft_messages(row)
        elif task == "smart_sft":
            messages = [
                {"role": "user", "content": f"Create a keyword mnemonic for: {row['term']}"},
                {"role": "assistant", "content": row["mnemonic"]},
            ]
        else:
            raise ValueError(f"unknown task: {task}")
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    ds = ds.map(render)
    train = cfg["training"]
    sft_cfg = SFTConfig(
        output_dir=cfg["output_dir"],
        learning_rate=train["learning_rate"],
        num_train_epochs=train["num_train_epochs"],
        per_device_train_batch_size=train["per_device_train_batch_size"],
        gradient_accumulation_steps=train.get("gradient_accumulation_steps", 1),
        warmup_ratio=train.get("warmup_ratio", 0.0),
        lr_scheduler_type=train.get("lr_scheduler_type", "cosine"),
        logging_steps=train.get("logging_steps", 10),
        save_strategy=train.get("save_strategy", "epoch"),
        eval_strategy="epoch" if "validation" in ds else "no",
        bf16=train.get("bf16", True),
        gradient_checkpointing=train.get("gradient_checkpointing", True),
        max_length=train.get("max_length", 1024),
        dataset_text_field="text",
        report_to=train.get("report_to", "none"),
        seed=seed,
    )
    trainer = SFTTrainer(
        model=model, args=sft_cfg, train_dataset=ds["train"],
        eval_dataset=ds.get("validation"), processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
