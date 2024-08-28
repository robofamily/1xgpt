import torch 
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
from dpt.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from magvit2.config import VQConfig
from magvit2.models.lfqgan import VQModel

class Attention(nn.Module):
    def __init__(self, ff_dim, head_dim, max_T, n_heads, drop_p, causal=False):
        # max_T: maximal sequence length
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_T = max_T
        self.causal = causal
        self.q_net = nn.Linear(ff_dim, head_dim * n_heads)
        self.k_net = nn.Linear(ff_dim, head_dim * n_heads)
        self.v_net = nn.Linear(ff_dim, head_dim * n_heads)
        self.proj_net = nn.Linear(head_dim * n_heads, ff_dim)
        self.drop_p = drop_p

    def forward(self, x):
        B, T, _ = x.shape # batch size, seq length, ff_dim
        E, D = self.n_heads, self.head_dim

        # Divide the tensors for multi head dot product
        q = self.q_net(x).view(B, T, E, D).transpose(1, 2) # b t (e d) -> b e t d
        k = self.k_net(x).view(B, T, E, D).transpose(1, 2) # b t (e d) -> b e t d
        v = self.v_net(x).view(B, T, E, D).transpose(1, 2) # b t (e d) -> b e t d

        inner = F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop_p, is_causal=self.causal)
        inner = inner.transpose(1, 2).contiguous().view(B, T, E * D) # b e t d -> b t (e d) Combine results from multi heads
        return self.proj_net(inner)

class Block(nn.Module):
    def __init__(self, ff_dim, head_dim, max_T, n_heads, drop_p, causal):
        super().__init__()
        self.ln1 = nn.LayerNorm(ff_dim)
        self.attn = Attention(ff_dim, head_dim, max_T, n_heads, drop_p, causal)
        self.ln2 = nn.LayerNorm(ff_dim)
        self.ff = nn.Sequential(
            nn.Linear(ff_dim, ff_dim * 4),
            nn.GELU(),
            nn.Linear(ff_dim * 4, ff_dim),
            nn.Dropout(drop_p),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, in_dim, out_dim, head_dim, n_heads, n_blocks, max_T, drop_p, causal):
        super().__init__()
        self.blocks = nn.ModuleList([Block(in_dim, head_dim, max_T, n_heads, drop_p, causal) for _ in range(n_blocks)])
        self.out_proj = nn.ModuleList([nn.Linear(in_dim, out_dim) for i in range(4)])
        devisor = 1 / torch.sqrt(torch.tensor(in_dim, dtype=torch.float32))
        self.pos_emb = nn.Parameter(torch.randn(1, max_T, in_dim) * devisor)
        self.apply(self._init_module)

    def _init_module(self, module):
        if isinstance(module, nn.Linear):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            constant_(module.bias, 0)
            constant_(module.weight, 1.0)

    def forward(self, x):
        B, T, C = x.shape # B: batch size, T: sequence length, C: token dim
        state = x + self.pos_emb[:, :T, :]
        hidden_states = []
        for block in self.blocks:
            state = block(state)
            hidden_states.append([state,])
        hidden_states = hidden_states[-4:]
        for i in range(len(hidden_states)):
            hidden_states[i][0] = self.out_proj[i](hidden_states[i][0])
        return hidden_states

class Magvit2Dpt(nn.Module):
    def __init__(
            self,
            freeze_dpt,
            n_blocks=8, 
            head_dim=8, 
            n_heads=64, 
            max_T=729, 
            drop_p=0.1,
            ):
        super().__init__()
        vq_config = VQConfig()
        self.magvit_model = VQModel(vq_config).to(dtype=torch.bfloat16)
        for param in self.magvit_model.parameters():
            param.requires_grad = False

        self.max_depth = 20 # take from original settings
        self.dpt_model = DepthAnythingV2(
            encoder='vitl',
            features=256,
            out_channels=[256, 512, 1024, 1024],
            max_depth=self.max_depth,
        ).depth_head
        self.freeze_dpt = freeze_dpt
            
        self.magvit2dpt = Transformer(
            in_dim=512, 
            out_dim=1024,
            head_dim=head_dim, 
            n_heads=n_heads, 
            n_blocks=n_blocks,
            max_T=max_T,
            drop_p=drop_p,
            causal=False,
        )
        print("number of parameters of magvit: {:e}".format(
            sum(p.numel() for p in self.magvit_model.parameters()))
        )
        print("number of parameters of dpt depth head: {:e}".format(
            sum(p.numel() for p in self.dpt_model.parameters()))
        )
        print("number of parameters of magvit2dpt: {:e}".format(
            sum(p.numel() for p in self.magvit2dpt.parameters()))
        )

    def load_magvit(self, magvit_ckpt_path):
        self.magvit_model.init_from_ckpt(magvit_ckpt_path)

    def load_dpt(self, dpt_ckpt_path):
        dpt_model = DepthAnythingV2(
            encoder='vitl',
            features=256,
            out_channels=[256, 512, 1024, 1024],
            max_depth=self.max_depth,
        )
        missing_keys, unexpected_keys = dpt_model.load_state_dict(torch.load(dpt_ckpt_path), strict=False)
        print('load ', dpt_ckpt_path, '\nmissing ', missing_keys, '\nunexpected ', unexpected_keys)
        self.dpt_model = dpt_model.depth_head
        if self.freeze_dpt:
            for param in self.dpt_model.parameters():
                param.requires_grad = False

    def decode(self, quant):
        features = self.magvit_model.decoder.conv_in(quant)
        for res in range(self.magvit_model.decoder.num_res_blocks):
            features = self.magvit_model.decoder.mid_block[res](features)

        B, C, H, W = features.shape
        features = features.to(dtype=torch.float32).flatten(2).permute(0, 2, 1) # (b c h w) -> (b hw c)
        features = self.magvit2dpt(features)

        depth = self.dpt_model(features, H, W) * self.max_depth
        return depth.squeeze(1)

    def forward(self, rgbs):
        quant, _, _, _ = self.magvit_model.encode(rgbs.to(dtype=torch.bfloat16))
        return self.decode(quant)
