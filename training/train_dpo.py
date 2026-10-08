#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from goro_llm.training_data import dpo_row


def main() -> None:
    p = argparse.ArgumentParser(description="DPO runner using TRL + PEFT")
    p.add_argument("config", type=Path)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import DPOConfig, DPOTrainer
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
    ds = load_dataset("json", data_files=cfg["data_files"]).map(dpo_row)
    lora = cfg["lora"]
    peft_config = LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora.get("dropout", 0.0),
        target_modules=lora["target_modules"], bias="none", task_type="CAUSAL_LM",
    )
    train = cfg["training"]
    dpo_cfg = DPOConfig(
        output_dir=cfg["output_dir"],
        beta=train.get("beta", 0.1),
        learning_rate=train["learning_rate"],
        num_train_epochs=train["num_train_epochs"],
        per_device_train_batch_size=train["per_device_train_batch_size"],
        gradient_accumulation_steps=train.get("gradient_accumulation_steps", 1),
        warmup_ratio=train.get("warmup_ratio", 0.0),
        lr_scheduler_type=train.get("lr_scheduler_type", "cosine"),
        bf16=train.get("bf16", True),
        gradient_checkpointing=train.get("gradient_checkpointing", True),
        max_length=train.get("max_length", 1024),
        logging_steps=train.get("logging_steps", 10),
        save_strategy=train.get("save_strategy", "epoch"),
        report_to=train.get("report_to", "none"),
        seed=int(cfg.get("seed", 42)),
    )
    trainer = DPOTrainer(
        model=model, args=dpo_cfg, train_dataset=ds["train"],
        eval_dataset=ds.get("validation"), processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(cfg["output_dir"])


if __name__ == "__main__":
    main()
