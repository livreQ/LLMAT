#!/usr/bin/env python
# -*- coding=utf8 -*-
"""
# Author: qichen@cs.toronto.ca
# Created Time : Thu Jan 16 19:28:09 2025
# File Name: base_llms.py
# Description:
"""

from src.algos.llmat.llms.molformer import MolFormer
from src.algos.llmat.llms.roberta import Roberta
from src.algos.llmat.llms.gpt2 import GPT2
from src.algos.llmat.llms.llama2 import Llama2
from src.algos.llmat.llms.t5 import T5
from src.utils.configs import LLMFeatureType
from peft import LoraConfig, get_peft_model


def get_model(foundation_model, token=False, lora_cfg=None):
    print("=============== get model ============")
    print("llm_model:" + foundation_model)
    if foundation_model == "molformer":
        model = MolFormer()
        target_modules = ["query", "value"]
    elif "roberta" in foundation_model:
        model = Roberta(kind=foundation_model, reduction=LLMFeatureType.AVERAGE)
        target_modules = ["query", "value"]
    elif "gpt2" in foundation_model:
        model = GPT2(kind=foundation_model, reduction=LLMFeatureType.AVERAGE)
        target_modules = ["c_attn"]
    elif "llama-2" in foundation_model:
        model = Llama2(kind=foundation_model, reduction=LLMFeatureType.AVERAGE)
        target_modules = ["q_proj", "v_proj"]
    elif "t5" in foundation_model:
        if "chem" in foundation_model:
            foundation_model_real = "GT4SD/multitask-text-and-chemistry-t5-base-augm"
            model = T5(kind=foundation_model_real, reduction=LLMFeatureType.AVERAGE)
        else:
            model = T5(kind=foundation_model, reduction=LLMFeatureType.AVERAGE)
        target_modules = ["q", "v"]
    elif foundation_model in ("fingerprints", "gin", "schnet"):
        model = MolFormer()
        target_modules = ["query", "value"]
    else:
        raise NotImplementedError

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=target_modules,
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        modules_to_save=["head"],
    )
    peft_model = get_peft_model(model, config)
    # for p in peft_model.base_model.head.original_module.parameters():
    #     p.requires_grad = False
    # for n, p in lora_model.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    if token:
        return peft_model, tokenizer
    else:
        return peft_model
