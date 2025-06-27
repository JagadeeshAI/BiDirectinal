import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed
from collections import OrderedDict
import torch
import copy


class Adapter_lora(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar=1.0,
                 adapter_layernorm_option="in"):
        super().__init__()

        # Dimensions
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        # Adapter scaling
        self.scalar = float(adapter_scalar)

        # Optional LayerNorm
        if adapter_layernorm_option == "in":
            self.layernorm = nn.LayerNorm(self.n_embd)
        else:
            self.layernorm = nn.Identity()

        # Projections
        self.lora_B = nn.Linear(self.n_embd, self.down_size, bias=False)  # Down
        self.lora_A = nn.Linear(self.down_size, self.n_embd, bias=False)  # Up
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
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., msa = [0,0,0]):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.ffn_option = 'parallel'
        self.msa = msa


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()


    def forward(self, x, adapt=None, prompt = None, rank_prompt = None, block_weight = None):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if adapt is not None:
            if block_weight is not None:
                block_weight = block_weight
            else:
                block_weight = torch.ones(3).cuda()
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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None, layer_id=None):
        super().__init__()
        self.config = config
        self.msa_adapt = True

        self.norm1 = norm_layer(dim)
        self.attn = Attention_lora(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                   attn_drop=attn_drop, proj_drop=drop, msa=config.msa)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        self.use_gslora = hasattr(config, "task_type") and ("gs" in config.task_type or "both" in config.task_type)


        if self.use_gslora:
            # Apply LoRA to fc1 and fc2 separately
            self.ffn_lora_fc1 = Adapter_lora(
                config=config,
                dropout=0.0,
                bottleneck=config.ffn_num,
                init_option=config.ffn_adapter_init_option,
                adapter_scalar=config.ffn_adapter_scalar,
                adapter_layernorm_option=config.ffn_adapter_layernorm_option
            )
            self.ffn_lora_fc2 = Adapter_lora(
                config=config,
                dropout=0.0,
                bottleneck=config.ffn_num,
                init_option=config.ffn_adapter_init_option,
                adapter_scalar=config.ffn_adapter_scalar,
                adapter_layernorm_option=config.ffn_adapter_layernorm_option
            )
            for p in self.ffn_lora_fc1.parameters():
                p.requires_grad = True
            for p in self.ffn_lora_fc2.parameters():
                p.requires_grad = True
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

            # Optional LoRA for fc1
            if self.ffn_lora_fc1 is not None:
                x = x + self.ffn_lora_fc1(x)

            x_fc1 = self.fc1(x)
            x_act = self.act(x_fc1)
            x_act = self.mlp_drop(x_act)

            x_fc2 = self.fc2(x_act)

            # ❗ Corrected: LoRA should act on the output of fc2, not the input
            if self.ffn_lora_fc2 is not None:
                x_fc2 = x_fc2 + self.ffn_lora_fc2(x_fc2)

            x_out = self.mlp_drop(x_fc2)
            x = residual + self.drop_path(x_out)

        return x



class VisionClassifier(nn.Module):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', tuning_config=None):
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
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
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

        self.adapt_pos = self.general_pos+ self.specfic_pos
        self.adapt_pos = sorted(self.adapt_pos)


        if self.use_distillation:
            self.old_adapter_list = nn.ModuleList()

        if self.use_block_weight:
            self.block_weight_list = []
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, .5, 1.5)


        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

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
                [nn.Parameter(torch.empty(1, self.tuning_config.vpt_num, embed_dim)) for _ in
                 range(depth)])
            for eee in self.embeddings:
                torch.nn.init.xavier_uniform_(eee.data)

        self.task_type = getattr(tuning_config, "task_type", "cl")


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
                        adapter_layernorm_option=config.ffn_adapter_layernorm_option
                    ).to(self._device)
                else:
                    adapter = nn.Identity()
                temp_adapter.append(adapter)

            self.cur_adapter.append(temp_adapter)

        self.cur_adapter.requires_grad_(True)
        print(f"✅ Initialized {len(self.adapt_pos)} adapter positions with {len(self.msa)} modules each.")


    def get_new_adapter_msa(self):
        config = self.config

        if config.ffn_adapt:
            for i in range(len(self.specfic_pos)):
                pos = self.adapt_pos.index(self.specfic_pos[i])
                temp_adapter = nn.ModuleList()
                for j in self.msa:
                    if j == 1:
                        adapter = Adapter_lora(self.config, dropout=0.0, bottleneck=config.ffn_num,
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
            temp_adapter.append(copy.deepcopy(self.cur_adapter[temp_pos].requires_grad_(False)))
        self.adapter_list.append(temp_adapter)

        if self.use_block_weight:
            self.block_weight_old = copy.deepcopy(self.block_weight)
            self.block_weight_list.append(self.block_weight_old.requires_grad_(False))
            self.block_weight = nn.Parameter(torch.randn(3, len(self.specfic_pos)))
            nn.init.uniform_(self.block_weight, .5, 1.5)
            print(self.block_weight_list)


        self.adapter_pos_list.append(self.adapt_pos)

        if self.use_distillation:
            self.old_adapter_list.append(copy.deepcopy(self.cur_adapter).requires_grad_(False))
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
                        x = blk(x, self.cur_adapter[pos], prompt, rank_prompt,
                                block_weight=self.block_weight[:, pos_spec])
                    else:
                        x = blk(x, self.cur_adapter[pos], prompt, rank_prompt, block_weight=None)
                else:
                    x = blk(x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            else:
                x = blk(x, adapt=None, prompt=prompt, rank_prompt=rank_prompt, block_weight=None)
            if self.config.vpt_on:
                x = x[:, self.config.vpt_num:, :]

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


    def forward(self, x, test=False, use_init_ptm=False):
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
            adapt = self.old_adapter_list[t_idx-1][pos]
            x_teacher = self.blocks[j](x_teacher, adapt)
        x_teacher = self.norm(x_teacher)
        output_teacher= x_teacher[:, 0, :]

        return output_new, output_teacher

    def get_current_block_weights(self):
        """
        Returns the current task's block weight matrix Ut of shape (3, num_specific_blocks),
        used for orthogonality loss across tasks.
        """
        if self.use_block_weight:
            return self.block_weight
        else:
            raise AttributeError("Block weight is not used in this configuration.")
