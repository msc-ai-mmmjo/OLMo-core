"""
Builds a HydraOLMo model with:
- config.heads_depth worth of layers in each Hydra head
- we always train with only 1 head (HydraLoRAConfig.n_heads is the number of *intended* final heads)
- LoRA params are only allowed in the truncated head
"""

import torch
from .config import HydraLoRAConfig
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import HydraTransformer, HydraTransformerConfig


def build_finetuning_model(config: HydraLoRAConfig) -> HydraTransformer:
    hydra_config = HydraTransformerConfig.from_olmo2_1B(
        n_heads=1, heads_depth=config.heads_depth
    )  # NOTE: in training we always initialise with only 1 head
    model = hydra_config.build(init_device="meta")

    # load model params
    hf_state = load_file(f"{config.weights_dir}/model.safetensors")
    olmo_state = convert_state_from_hf(None, hf_state)

    # load model state into hydra
    HydraTransformer.load_olmo_state(
        model, olmo_state, trunk_layers=hydra_config.trunk_layers, vocab_size=config.vocab_size
    )
    del hf_state, olmo_state
    model.to(device=config.device, dtype=torch.bfloat16)  # NOTE: param precision

    # inject LoRA into target modules specified by config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # replace target head with LoRA tuner head, always train with only 1 head
    model.heads[0] = get_peft_model(model.heads[0], lora_config)

    # all params except LoRA params are frozen
    model.requires_grad_(False)
    for n, p in model.named_parameters():
        if "lora" in n:
            p.requires_grad = True

    return model
