import torch
from functools import partial
from timm.layers import PatchEmbed
from vit_bid import VisionFace, VisionDectector  # Make sure both are imported


def print_parameter_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_m = total / 1e6
    trainable_m = trainable / 1e6
    percent = 100 * trainable / total if total > 0 else 0
    print(f"\n📊 Parameters: {total_m:.2f}M total | {trainable_m:.2f}M trainable ({percent:.2f}%)")



def get_model(use_lora=False, msa=[1, 0, 1], model_type="face"):
    class Args:
        def __init__(self):
            self.use_lora = use_lora
            self.msa = msa
            self.ffn_adapt = True
            self.msa_adapt = True
            self.vpt_on = False
            self.vpt_num = 0
            self.general_pos = [0, 1, 2, 3, 4, 5]
            self.specfic_pos = [6, 7, 8, 9, 10, 11]
            self.use_distillation = False
            self.use_block_weight = True
            self.ffn_num = 8
            self.ffn_adapter_init_option = "lora"
            self.ffn_adapter_scalar = "1.0"
            self.ffn_adapter_layernorm_option = "in"
            self.d_model = 768
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = Args()

    model_cls = VisionFace if model_type == "face" else VisionDectector

    model = model_cls(
        img_size=224 if model_type == "face" else 224,
        patch_size=16,
        embed_dim=768,
        num_classes=100,  # ✅ Add this line
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        tuning_config=args,
        embed_layer=PatchEmbed,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
    ).to(args._device)

    model=freeze_train(model)

    return model, args


def freeze_train(model):
    """
    Freezes classification logits for class indices 50–99.
    Keeps everything else trainable for training only on classes 0–49.
    """
    # Freeze only the classifier weights for class 50–99
    if hasattr(model, 'det_head') and isinstance(model.det_head, torch.nn.Linear):
        with torch.no_grad():
            # Zero gradients explicitly to avoid updates
            model.det_head.weight[50:].requires_grad = False
            model.det_head.bias[50:].requires_grad = False

    # Optionally: you can also print what's being frozen
    print("🔒 Freezing classification head weights for classes 50–99")

    return model


def apply_lora(model, args):
    for p in model.parameters():
        p.requires_grad = False

    if not args.use_lora:
        print("🚫 LoRA disabled — using full fine-tuning.")
        for p in model.parameters():
            p.requires_grad = True
        return model

    print("✅ Applying BiD-LoRA (CL + GS unified)")

    for adapter in model.cur_adapter:
        for module in adapter:
            if hasattr(module, "lora_A"):
                module.lora_A.weight.requires_grad = True
            if hasattr(module, "lora_B"):
                module.lora_B.weight.requires_grad = True

    for block in model.blocks:
        if hasattr(block, "ffn_lora_fc1") and block.ffn_lora_fc1 is not None:
            for p in block.ffn_lora_fc1.parameters():
                p.requires_grad = True
        if hasattr(block, "ffn_lora_fc2") and block.ffn_lora_fc2 is not None:
            for p in block.ffn_lora_fc2.parameters():
                p.requires_grad = True

    return model
