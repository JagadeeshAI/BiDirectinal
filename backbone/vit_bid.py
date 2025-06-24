import math
import torch
import torch.nn as nn
from timm.layers import DropPath
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.layers import PatchEmbed
from collections import OrderedDict
import torch
import copy
import torch.nn.functional as F


class Adapter_lora(nn.Module):
    def __init__(
        self,
        config=None,
        d_model=None,
        bottleneck=None,
        dropout=0.0,
        init_option="lora",
        adapter_scalar=1.0,
        adapter_layernorm_option="in",
    ):
        super().__init__()

        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.scalar = float(adapter_scalar)

        if adapter_layernorm_option == "in":
            self.layernorm = nn.LayerNorm(self.n_embd)
        else:
            self.layernorm = nn.Identity()

        self.lora_B = nn.Linear(self.n_embd, self.down_size, bias=False)
        self.lora_A = nn.Linear(self.down_size, self.n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Init
        if hasattr(config, "random_orth") and config.random_orth:
            rand_matrix = torch.rand(self.n_embd, self.down_size)
            q, _ = torch.linalg.qr(rand_matrix)
            with torch.no_grad():
                self.lora_B.weight.copy_(q.T)
        else:
            nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))

        if init_option == "lora":
            nn.init.zeros_(self.lora_A.weight)
        else:
            raise NotImplementedError(f"Init option {init_option} not supported.")

    def forward(self, x):
        x = self.layernorm(x)
        x = self.dropout(self.lora_B(x))
        x = self.lora_A(x)
        return self.scalar * x


class Attention_lora(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        msa=[0, 0, 0],
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.ffn_option = "parallel"
        self.msa = msa

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(self, x, adapt=None, prompt=None, rank_prompt=None, block_weight=None):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if adapt is not None:
            if block_weight is not None:
                block_weight = block_weight
            else:
                block_weight = torch.ones(3, device=x.device)

            if self.msa[0] == 1:
                adapt_x = adapt[0](x)
                q += block_weight[0] * adapt_x
            if self.msa[1] == 1:
                adapt_x = adapt[1](x)
                k += block_weight[1] * adapt_x
            if self.msa[2] == 1:
                adapt_x = adapt[2](x)
                v += block_weight[2] * adapt_x

        k = self._shape(k, -1, B).view(B * self.num_heads, -1, self.head_dim)
        v = self._shape(v, -1, B).view(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).view(B * self.num_heads, -1, self.head_dim)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        config=None,
        layer_id=None,
    ):
        super().__init__()
        self.config = config
        self.msa_adapt = True

        self.norm1 = norm_layer(dim)
        self.attn = Attention_lora(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            msa=config.msa,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        # 🔧 FFN LoRA (GS-style): activate based on config.ffn_adapt
        if config.ffn_adapt:
            self.ffn_lora_fc1 = Adapter_lora(
                config=config,
                dropout=0.0,
                bottleneck=config.ffn_num,
                init_option=config.ffn_adapter_init_option,
                adapter_scalar=config.ffn_adapter_scalar,
                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
            )
            self.ffn_lora_fc2 = Adapter_lora(
                config=config,
                dropout=0.0,
                bottleneck=config.ffn_num,
                init_option=config.ffn_adapter_init_option,
                adapter_scalar=config.ffn_adapter_scalar,
                adapter_layernorm_option=config.ffn_adapter_layernorm_option,
            )
        else:
            self.ffn_lora_fc1 = None
            self.ffn_lora_fc2 = None

    def forward(self, x, adapt=None, prompt=None, rank_prompt=None, block_weight=None):
        if self.msa_adapt:
            x = x + self.drop_path(
                self.attn(self.norm1(x), adapt, prompt, rank_prompt, block_weight)
            )

            residual = x
            x = self.norm2(x)

            # Apply FFN LoRA to input if present
            if self.ffn_lora_fc1 is not None:
                x = x + self.ffn_lora_fc1(x)

            x_fc1 = self.fc1(x)
            x_act = self.act(x_fc1)
            x_act = self.mlp_drop(x_act)

            x_fc2 = self.fc2(x_act)

            # Apply FFN LoRA to output if present
            if self.ffn_lora_fc2 is not None:
                x_fc2 = x_fc2 + self.ffn_lora_fc2(x_fc2)

            x_out = self.mlp_drop(x_fc2)
            x = residual + self.drop_path(x_out)

        return x


class VisionClassifier(nn.Module):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        global_pool=False,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        tuning_config=None,
    ):
        super().__init__()

        self.tuning_config = tuning_config
        self.config = tuning_config  # ✅ This line fixes the AttributeError

        if self.tuning_config.ffn_adapt:
            print("I'm using ViT with adapters.")
        else:
            print("I'm using ViT without adapters.")
            self.maskout_block = []
        self.adapt_msa = True
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.msa_adapt = self.tuning_config.msa_adapt
        self.use_distillation = self.tuning_config.use_distillation
        self.use_block_weight = self.tuning_config.use_block_weight

        if self.msa_adapt:
            self.msa = self.tuning_config.msa
        self.general_pos = self.tuning_config.general_pos
        self.specfic_pos = self.tuning_config.specfic_pos

        self.adapt_pos = self.general_pos + self.specfic_pos
        self.adapt_pos = sorted(self.adapt_pos)

        if self.use_distillation:
            self.old_adapter_list = nn.ModuleList()

        if self.use_block_weight:
            self.block_weight_list = []
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, 0.5, 1.5)

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    config=tuning_config,
                    layer_id=i,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict(
                    [
                        ("fc", nn.Linear(embed_dim, representation_size)),
                        ("act", nn.Tanh()),
                    ]
                )
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.head_dist = None
        if distilled:
            self.head_dist = (
                nn.Linear(self.embed_dim, self.num_classes)
                if num_classes > 0
                else nn.Identity()
            )

        ######### MAE begins ############
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        ######## Adapter begins #########
        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            # properly registered
            self.embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
                [
                    nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim))
                    for _ in range(depth)
                ]
            )
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.task_type = getattr(tuning_config, "task_type", "incremental_learning")

        self._device = tuning_config._device
        self.adapter_list = []
        self.adapter_pos_list = []
        self.cur_adapter = nn.ModuleList()
        if self.msa_adapt:
            self.get_new_adapter_initial_msa()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True

    def get_new_adapter_initial_msa(self):
        config = self.config

        if not getattr(self, "use_lora", True):
            print("🚫 LoRA is disabled. Skipping adapter initialization.")
            return

        if not config.ffn_adapt:
            print("🚫 FFN adaptation is disabled. Skipping adapter initialization.")
            return

        for i in range(len(self.adapt_pos)):
            temp_adapter = nn.ModuleList()
            for j in self.msa:
                if j == 1:
                    adapter = Adapter_lora(
                        config=config,
                        dropout=0.0,
                        bottleneck=config.ffn_num,
                        init_option=config.ffn_adapter_init_option,
                        adapter_scalar=config.ffn_adapter_scalar,
                        adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                    ).to(self._device)
                else:
                    adapter = nn.Identity()
                temp_adapter.append(adapter)

            self.cur_adapter.append(temp_adapter)

        self.cur_adapter.requires_grad_(True)
        print(
            f"✅ Initialized {len(self.adapt_pos)} adapter positions with {len(self.msa)} modules each."
        )

    def get_new_adapter_msa(self):
        config = self.config

        if config.ffn_adapt:
            for i in range(len(self.specfic_pos)):
                pos = self.adapt_pos.index(self.specfic_pos[i])
                temp_adapter = nn.ModuleList()
                for j in self.msa:
                    if j == 1:
                        adapter = Adapter_lora(
                            self.config,
                            dropout=0.0,
                            bottleneck=config.ffn_num,
                            init_option=config.ffn_adapter_init_option,
                            adapter_scalar=config.ffn_adapter_scalar,
                            adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                        ).to(self._device)
                        adapter.requires_grad_(True)
                    else:
                        adapter = nn.Identity()
                    temp_adapter.append(adapter)
                self.cur_adapter[pos] = temp_adapter

            if len(self.specfic_pos) < 12:
                self.cur_adapter.requires_grad_(True)

                for i in self.adapt_pos:
                    if i in self.general_pos:
                        pos = self.adapt_pos.index(i)
                        for j in range(len(self.msa)):
                            if self.msa[j] == 1:
                                self.cur_adapter[pos][j].lora_B.requires_grad_(False)
        else:
            print("====Not use adapter===")

    def add_adapter_to_list(self):
        temp_adapter = []
        for i in range(len(self.specfic_pos)):
            temp_pos = self.adapt_pos.index(self.specfic_pos[i])
            temp_adapter.append(
                copy.deepcopy(self.cur_adapter[temp_pos].requires_grad_(False))
            )
        self.adapter_list.append(temp_adapter)

        if self.use_block_weight:
            self.block_weight_old = copy.deepcopy(self.block_weight)
            self.block_weight_list.append(self.block_weight_old.requires_grad_(False))
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, 0.5, 1.5)
            print(self.block_weight_list)

        self.adapter_pos_list.append(self.adapt_pos)

        if self.use_distillation:
            self.old_adapter_list.append(
                copy.deepcopy(self.cur_adapter).requires_grad_(False)
            )
        if self.msa_adapt:
            self.get_new_adapter_msa()

    def forward_train(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            rank_prompt = None
            prompt = None

            if self.config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)

            if self.config.ffn_adapt:
                if idx in self.adapt_pos:
                    pos = self.adapt_pos.index(idx)
                    block_weight = None
                    if self.use_block_weight and idx in self.specfic_pos:
                        pos_spec = self.specfic_pos.index(idx)
                        x = blk(
                            x,
                            self.cur_adapter[pos],
                            prompt,
                            rank_prompt,
                            block_weight=self.block_weight[:, pos_spec],
                        )
                    else:
                        x = blk(
                            x,
                            self.cur_adapter[pos],
                            prompt,
                            rank_prompt,
                            block_weight=None,
                        )
                else:
                    x = blk(
                        x,
                        adapt=None,
                        prompt=prompt,
                        rank_prompt=rank_prompt,
                        block_weight=None,
                    )
            else:
                x = blk(
                    x,
                    adapt=None,
                    prompt=prompt,
                    rank_prompt=rank_prompt,
                    block_weight=None,
                )
            if self.config.vpt_on:
                x = x[:, self.config.vpt_num :, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward(self, x):
        return self.forward_train(x)

    def forward_general_cls(self, x, t_idx):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_teacher = copy.deepcopy(x)

        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.cur_adapter[pos]
            x = self.blocks[j](x, adapt)

        x = self.norm(x)
        output_new = x[:, 0, :]

        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.old_adapter_list[t_idx - 1][pos]
            x_teacher = self.blocks[j](x_teacher, adapt)
        x_teacher = self.norm(x_teacher)
        output_teacher = x_teacher[:, 0, :]

        return output_new, output_teacher


class VisionFace(nn.Module):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        global_pool=False,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        tuning_config=None,
    ):
        super().__init__()

        self.tuning_config = tuning_config
        self.config = tuning_config

        if self.tuning_config.ffn_adapt:
            print("I'm using ViT with adapters.")
        else:
            print("I'm using ViT without adapters.")
            self.maskout_block = []
        self.adapt_msa = True
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.msa_adapt = self.tuning_config.msa_adapt
        self.use_distillation = self.tuning_config.use_distillation
        self.use_block_weight = self.tuning_config.use_block_weight

        if self.msa_adapt:
            self.msa = self.tuning_config.msa
        self.general_pos = self.tuning_config.general_pos
        self.specfic_pos = self.tuning_config.specfic_pos

        self.adapt_pos = self.general_pos + self.specfic_pos
        self.adapt_pos = sorted(self.adapt_pos)

        if self.use_distillation:
            self.old_adapter_list = nn.ModuleList()

        if self.use_block_weight:
            self.block_weight_list = []
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, 0.5, 1.5)

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    config=tuning_config,
                    layer_id=i,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm

        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            self.embeddings = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim))
                    for _ in range(depth)
                ]
            )
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.task_type = getattr(tuning_config, "task_type", "incremental_learning")

        self._device = tuning_config._device
        self.adapter_list = []
        self.adapter_pos_list = []
        self.cur_adapter = nn.ModuleList()
        if self.msa_adapt:
            self.get_new_adapter_initial_msa()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True

    def get_new_adapter_initial_msa(self):
        config = self.config

        if not getattr(self, "use_lora", True):
            print("🚫 LoRA is disabled. Skipping adapter initialization.")
            return

        if not config.ffn_adapt:
            print("🚫 FFN adaptation is disabled. Skipping adapter initialization.")
            return

        for i in range(len(self.adapt_pos)):
            temp_adapter = nn.ModuleList()
            for j in self.msa:
                if j == 1:
                    adapter = Adapter_lora(
                        config=config,
                        dropout=0.0,
                        bottleneck=config.ffn_num,
                        init_option=config.ffn_adapter_init_option,
                        adapter_scalar=config.ffn_adapter_scalar,
                        adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                    ).to(self._device)
                else:
                    adapter = nn.Identity()
                temp_adapter.append(adapter)

            self.cur_adapter.append(temp_adapter)

        self.cur_adapter.requires_grad_(True)
        print(
            f"✅ Initialized {len(self.adapt_pos)} adapter positions with {len(self.msa)} modules each."
        )

    def get_new_adapter_msa(self):
        config = self.config

        if config.ffn_adapt:
            for i in range(len(self.specfic_pos)):
                pos = self.adapt_pos.index(self.specfic_pos[i])
                temp_adapter = nn.ModuleList()
                for j in self.msa:
                    if j == 1:
                        adapter = Adapter_lora(
                            self.config,
                            dropout=0.0,
                            bottleneck=config.ffn_num,
                            init_option=config.ffn_adapter_init_option,
                            adapter_scalar=config.ffn_adapter_scalar,
                            adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                        ).to(self._device)
                        adapter.requires_grad_(True)
                    else:
                        adapter = nn.Identity()
                    temp_adapter.append(adapter)
                self.cur_adapter[pos] = temp_adapter

            if len(self.specfic_pos) < 12:
                self.cur_adapter.requires_grad_(True)

                for i in self.adapt_pos:
                    if i in self.general_pos:
                        pos = self.adapt_pos.index(i)
                        for j in range(len(self.msa)):
                            if self.msa[j] == 1:
                                self.cur_adapter[pos][j].lora_B.requires_grad_(False)
        else:
            print("====Not use adapter===")

    def add_adapter_to_list(self):
        temp_adapter = []
        for i in range(len(self.specfic_pos)):
            temp_pos = self.adapt_pos.index(self.specfic_pos[i])
            temp_adapter.append(
                copy.deepcopy(self.cur_adapter[temp_pos].requires_grad_(False))
            )
        self.adapter_list.append(temp_adapter)

        if self.use_block_weight:
            self.block_weight_old = copy.deepcopy(self.block_weight)
            self.block_weight_list.append(self.block_weight_old.requires_grad_(False))
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, 0.5, 1.5)
            print(self.block_weight_list)

        self.adapter_pos_list.append(self.adapt_pos)

        if self.use_distillation:
            self.old_adapter_list.append(
                copy.deepcopy(self.cur_adapter).requires_grad_(False)
            )
        if self.msa_adapt:
            self.get_new_adapter_msa()

    def forward_train(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            rank_prompt = None
            prompt = None

            if self.config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)

            if self.config.ffn_adapt:
                if idx in self.adapt_pos:
                    pos = self.adapt_pos.index(idx)
                    block_weight = None
                    if self.use_block_weight and idx in self.specfic_pos:
                        pos_spec = self.specfic_pos.index(idx)
                        x = blk(
                            x,
                            self.cur_adapter[pos],
                            prompt,
                            rank_prompt,
                            block_weight=self.block_weight[:, pos_spec],
                        )
                    else:
                        x = blk(
                            x,
                            self.cur_adapter[pos],
                            prompt,
                            rank_prompt,
                            block_weight=None,
                        )
                else:
                    x = blk(
                        x,
                        adapt=None,
                        prompt=prompt,
                        rank_prompt=rank_prompt,
                        block_weight=None,
                    )
            else:
                x = blk(
                    x,
                    adapt=None,
                    prompt=prompt,
                    rank_prompt=rank_prompt,
                    block_weight=None,
                )
            if self.config.vpt_on:
                x = x[:, self.config.vpt_num :, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]

        embedding = F.normalize(x, p=2, dim=1)
        return embedding

    def forward(self, x):
        return self.forward_train(x)

    def forward_general_cls(self, x, t_idx):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_teacher = copy.deepcopy(x)

        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.cur_adapter[pos]
            x = self.blocks[j](x, adapt)

        x = self.norm(x)
        output_new = x[:, 0, :]

        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.old_adapter_list[t_idx - 1][pos]
            x_teacher = self.blocks[j](x_teacher, adapt)
        x_teacher = self.norm(x_teacher)
        output_teacher = x_teacher[:, 0, :]

        return output_new, output_teacher



class VisionDectector(nn.Module):
    """Vision Transformer with support for global average pooling"""

    def __init__(
        self,
        global_pool=False,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=100,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        tuning_config=None,
    ):
        super().__init__()

        self.tuning_config = tuning_config
        self.config = tuning_config

        self.num_classes = num_classes

        if self.tuning_config.ffn_adapt:
            print("I'm using ViT with adapters.")
        else:
            print("I'm using ViT without adapters.")
            self.maskout_block = []
        self.adapt_msa = True
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.msa_adapt = self.tuning_config.msa_adapt
        self.use_distillation = self.tuning_config.use_distillation
        self.use_block_weight = self.tuning_config.use_block_weight

        if self.msa_adapt:
            self.msa = self.tuning_config.msa
        self.general_pos = self.tuning_config.general_pos
        self.specfic_pos = self.tuning_config.specfic_pos

        self.adapt_pos = self.general_pos + self.specfic_pos
        self.adapt_pos = sorted(self.adapt_pos)

        if self.use_distillation:
            self.old_adapter_list = nn.ModuleList()

        if self.use_block_weight:
            self.block_weight_list = []
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, 0.5, 1.5)

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = (
            nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tokens, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    config=tuning_config,
                    layer_id=i,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

            del self.norm

        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0, tuning_config.vpt_num
            self.embeddings = nn.ParameterList(
                [
                    nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim))
                    for _ in range(depth)
                ]
            )
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.task_type = getattr(tuning_config, "task_type", "incremental_learning")

        self._device = tuning_config._device
        self.adapter_list = []
        self.adapter_pos_list = []
        self.cur_adapter = nn.ModuleList()
        if self.msa_adapt:
            self.get_new_adapter_initial_msa()
        self.num_queries = 100  # or 300, depending on your GPU
        self.det_head = nn.Linear(embed_dim, 4 + self.num_classes)



    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

        for i in range(len(self.cur_adapter)):
            self.cur_adapter[i].requires_grad = True

    def get_new_adapter_initial_msa(self):
        config = self.config

        if not getattr(self, "use_lora", True):
            print("🚫 LoRA is disabled. Skipping adapter initialization.")
            return

        if not config.ffn_adapt:
            print("🚫 FFN adaptation is disabled. Skipping adapter initialization.")
            return

        for i in range(len(self.adapt_pos)):
            temp_adapter = nn.ModuleList()
            for j in self.msa:
                if j == 1:
                    adapter = Adapter_lora(
                        config=config,
                        dropout=0.0,
                        bottleneck=config.ffn_num,
                        init_option=config.ffn_adapter_init_option,
                        adapter_scalar=config.ffn_adapter_scalar,
                        adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                    ).to(self._device)
                else:
                    adapter = nn.Identity()
                temp_adapter.append(adapter)

            self.cur_adapter.append(temp_adapter)

        self.cur_adapter.requires_grad_(True)
        print(
            f"✅ Initialized {len(self.adapt_pos)} adapter positions with {len(self.msa)} modules each."
        )

    def get_new_adapter_msa(self):
        config = self.config

        if config.ffn_adapt:
            for i in range(len(self.specfic_pos)):
                pos = self.adapt_pos.index(self.specfic_pos[i])
                temp_adapter = nn.ModuleList()
                for j in self.msa:
                    if j == 1:
                        adapter = Adapter_lora(
                            self.config,
                            dropout=0.0,
                            bottleneck=config.ffn_num,
                            init_option=config.ffn_adapter_init_option,
                            adapter_scalar=config.ffn_adapter_scalar,
                            adapter_layernorm_option=config.ffn_adapter_layernorm_option,
                        ).to(self._device)
                        adapter.requires_grad_(True)
                    else:
                        adapter = nn.Identity()
                    temp_adapter.append(adapter)
                self.cur_adapter[pos] = temp_adapter

            if len(self.specfic_pos) < 12:
                self.cur_adapter.requires_grad_(True)

                for i in self.adapt_pos:
                    if i in self.general_pos:
                        pos = self.adapt_pos.index(i)
                        for j in range(len(self.msa)):
                            if self.msa[j] == 1:
                                self.cur_adapter[pos][j].lora_B.requires_grad_(False)
        else:
            print("====Not use adapter===")

    def add_adapter_to_list(self):
        temp_adapter = []
        for i in range(len(self.specfic_pos)):
            temp_pos = self.adapt_pos.index(self.specfic_pos[i])
            temp_adapter.append(
                copy.deepcopy(self.cur_adapter[temp_pos].requires_grad_(False))
            )
        self.adapter_list.append(temp_adapter)

        if self.use_block_weight:
            self.block_weight_old = copy.deepcopy(self.block_weight)
            self.block_weight_list.append(self.block_weight_old.requires_grad_(False))
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, 0.5, 1.5)
            print(self.block_weight_list)

        self.adapter_pos_list.append(self.adapt_pos)

        if self.use_distillation:
            self.old_adapter_list.append(
                copy.deepcopy(self.cur_adapter).requires_grad_(False)
            )
        if self.msa_adapt:
            self.get_new_adapter_msa()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for idx, blk in enumerate(self.blocks):
            rank_prompt = None
            prompt = None

            if self.config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = torch.cat([eee, x], dim=1)

            if self.config.ffn_adapt:
                if idx in self.adapt_pos:
                    pos = self.adapt_pos.index(idx)
                    if self.use_block_weight and idx in self.specfic_pos:
                        pos_spec = self.specfic_pos.index(idx)
                        x = blk(
                            x,
                            self.cur_adapter[pos],
                            prompt,
                            rank_prompt,
                            block_weight=self.block_weight[:, pos_spec],
                        )
                    else:
                        x = blk(
                            x,
                            self.cur_adapter[pos],
                            prompt,
                            rank_prompt,
                            block_weight=None,
                        )
                else:
                    x = blk(
                        x,
                        adapt=None,
                        prompt=prompt,
                        rank_prompt=rank_prompt,
                        block_weight=None,
                    )
            else:
                x = blk(
                    x,
                    adapt=None,
                    prompt=prompt,
                    rank_prompt=rank_prompt,
                    block_weight=None,
                )

            if self.config.vpt_on:
                x = x[:, self.config.vpt_num :, :]

        x = self.norm(x)
        x = x[:, 1:, :]
        return x 


    def forward(self, images, targets=None):
        x = torch.stack(images, dim=0)  # [B, 3, 224, 224]
        B = x.shape[0]

        # Feature extraction
        features = self.forward_features(x)         # [B, N_patches, D]
        pooled = features.mean(dim=1)               # [B, D]

        # Prediction head
        preds = self.det_head(pooled)  # [B, 4 + num_classes]
        bbox_preds = preds[:, :4]      # [B, 4]
        class_logits = preds[:, 4:]    # [B, num_classes]


        # Normalize predicted bboxes to image scale
        bbox_preds = torch.sigmoid(bbox_preds) * 224.0

        if self.training and targets is not None:
            # Build lists of GT boxes and labels
            all_gt_boxes = []
            all_gt_labels = []
            for t in targets:
                gt_boxes = t["boxes"].to(preds.device)  # [n, 4]
                gt_labels = t["labels"].to(preds.device)  # [n]

                # Convert normalized GT boxes to absolute scale
                if gt_boxes.max() <= 1.0:
                    gt_boxes = gt_boxes * 224.0

                all_gt_boxes.append(gt_boxes)
                all_gt_labels.append(gt_labels)

            # Flatten predictions (Q queries per B batch)
            pred_boxes = bbox_preds.view(-1, 4)             # [B*Q, 4]
            pred_logits = class_logits.view(-1, self.num_classes)  # [B*Q, C]

            # Flatten targets
            tgt_boxes = torch.cat(all_gt_boxes, dim=0)      # [∑n, 4]
            tgt_labels = torch.cat(all_gt_labels, dim=0)    # [∑n]

            # Losses (simple, without Hungarian matching)
            loss_bbox = F.smooth_l1_loss(pred_boxes[:tgt_boxes.size(0)], tgt_boxes)
            loss_cls = F.cross_entropy(pred_logits[:tgt_labels.size(0)], tgt_labels)

            return {"loss_cls": loss_cls, "loss_bbox": loss_bbox}

        else:
            # Inference mode — format outputs for each image
            outputs = []
            for i in range(B):
                scores = torch.softmax(class_logits[i], dim=-1)        # [Q, num_classes]
                labels = torch.argmax(scores, dim=-1)                  # [Q]
                outputs.append({
                    "boxes": bbox_preds[i],                            # [Q, 4]
                    "scores": scores.max(dim=-1).values,               # [Q]
                    "labels": labels                                   # [Q]
                })

            return outputs


    def forward_general_cls(self, x, t_idx):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_teacher = copy.deepcopy(x)

        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.cur_adapter[pos]
            x = self.blocks[j](x, adapt)

        x = self.norm(x)
        output_new = x[:, 0, :]

        for j in self.general_pos:
            pos = self.adapt_pos.index(j)
            adapt = self.old_adapter_list[t_idx - 1][pos]
            x_teacher = self.blocks[j](x_teacher, adapt)
        x_teacher = self.norm(x_teacher)
        output_teacher = x_teacher[:, 0, :]

        return output_new, output_teacher
