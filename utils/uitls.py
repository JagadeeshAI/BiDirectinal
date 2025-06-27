import torch
from functools import partial
from backbone.vit_bid import VisionClassifier

def print_parameter_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_m = total / 1e6
    trainable_m = trainable / 1e6
    percent = 100 * trainable / total if total > 0 else 0
    print(f"\n📊 Parameters: {total_m:.2f}M total | {trainable_m:.2f}M trainable ({percent:.2f}%)")

def get_model(class_range,use_lora=False, msa=[1, 0, 1]):
    class Args:
        def __init__(self):
            self.task_type = "cl"                     # GS-LoRA mode
            self.use_lora = True                      # Enable LoRA (even if from scratch)
            self.ffn_adapt = True                     # Enable FFN adaptation (GS-LoRA)
            
            self.vpt_on = False                       # Not using VPT
            self.vpt_num = 0                          # No VPT prompts

            self.msa = [1, 0, 1]                      # Apply LoRA to first and last MSA layers
            self.general_pos = [0, 1, 2, 3, 4, 5]     # Shared adapter positions
            self.specfic_pos = [6, 7, 8, 9, 10, 11]   # Task-specific adapter positions

            self.use_distillation = True              # Enable distillation (used in original paper)
            self.use_block_weight = True              # Enable block weighting mechanism

            self.ffn_num = 8                          # Adapter bottleneck size (rank)
            self.ffn_adapter_init_option = "lora"     # Use zero-init (residual style)
            self.ffn_adapter_scalar = "1.0"           # Scaling factor (fixed or learnable)
            self.ffn_adapter_layernorm_option = "in"  # LayerNorm inside adapter

            self.d_model = 768                        # ViT-Base hidden dimension
            self.msa_adapt = True                     # Enable MSA adaptation (for CL-LoRA)
            
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = Args()

    img_size = 224

    model = VisionClassifier(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        tuning_config=args,
    ).to(args._device)

    # if not use_lora:
        # model = freeze_output(model,class_range)
    print_parameter_stats(model)

    return model, args

def freeze_output(model, class_range, head_attr="head"):
    """
    Freezes classifier head weights for output nodes outside class_range.
    Keeps everything else trainable.
    
    Args:
        model: Your model (must have an attribute for classifier head, e.g., model.head or model.det_head).
        class_range: Tuple (start_class, end_class), inclusive, specifying trainable class indices.
        head_attr: Name of classifier head attribute ("head" by default, "det_head" for detection head).
    Returns:
        Model with correct parameters frozen.
    """
    head = getattr(model, head_attr, None)
    if not (isinstance(head, torch.nn.Linear)):
        print(f"❌ Could not find valid classifier head '{head_attr}' in model.")
        return model

    num_outputs = head.weight.shape[0]
    start_class, end_class = class_range
    frozen_indices = list(range(0, start_class)) + list(range(end_class + 1, num_outputs))
    trainable_indices = list(range(start_class, end_class + 1))

    print(f"🔒 Freezing classifier head weights for classes: {frozen_indices}")

    def partial_freeze_hook(grad):
        grad = grad.clone()  
        if len(frozen_indices) > 0:
            grad[frozen_indices] = 0
        return grad

    head.weight.register_hook(partial_freeze_hook)
    head.bias.register_hook(partial_freeze_hook)

    return model


def apply_lora(model, args):
    for p in model.parameters():
        p.requires_grad = False

    if not args.use_lora:
        print("🚫 LoRA disabled — using full fine-tuning.")
        for p in model.parameters():
            p.requires_grad = True
        return model

    print("✅ Applying BiD-LoRA")

    for adapter in model.cur_adapter:
        for module in adapter:
            if hasattr(module, "lora_A"):
                module.lora_A.weight.requires_grad = True
            if hasattr(module, "lora_B"):
                module.lora_B.weight.requires_grad = True

    # for block in model.blocks:
    #     if hasattr(block, "ffn_lora_fc1") and block.ffn_lora_fc1 is not None:
    #         for p in block.ffn_lora_fc1.parameters():
    #             p.requires_grad = True
    #     if hasattr(block, "ffn_lora_fc2") and block.ffn_lora_fc2 is not None:
    #         for p in block.ffn_lora_fc2.parameters():
    #             p.requires_grad = True

    return model