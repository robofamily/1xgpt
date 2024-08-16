from torch import nn, Tensor
from einops import rearrange, repeat

from genie.attention import SelfAttention


class Mlp(nn.Module):
    def __init__(
        self,
        d_model: int,
        mlp_ratio: float = 4.0,
        mlp_bias: bool = True,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=mlp_bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model, bias=mlp_bias)
        self.drop = nn.Dropout(mlp_drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class STBlock(nn.Module):
    # See Figure 4 of https://arxiv.org/pdf/2402.15391.pdf
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_bias: bool = True,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        # sequence dim is over each frame's 16x16 patch tokens
        self.spatial_attn = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
        )

        # sequence dim is over time sequence (16)
        self.temporal_attn = SelfAttention(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
        )
        
        self.norm2 = nn.Identity() if qk_norm else nn.LayerNorm(d_model, eps=1e-05)
        self.mlp = Mlp(d_model=d_model, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, mlp_drop=mlp_drop)

        self.spatial_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model, bias=True)
        )

        self.temporal_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model, bias=True)
        )

        self.mlp_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model, bias=True)
        )
        
    def forward(self, x_TSC: Tensor, cond: Tensor) -> Tensor:
        # Process attention spatially
        T, S = x_TSC.size(1), x_TSC.size(2)
        x_SC = rearrange(x_TSC, 'B T S C -> (B T) S C')
        spatial_shift, spatial_scale, spatial_gate = repeat(self.spatial_adaLN(cond), 'B C -> (B T) 1 C', T=T).chunk(3, dim=-1)
        x_SC = x_SC + spatial_gate * self.spatial_attn(modulate(self.norm1(x_SC), spatial_shift, spatial_scale))

        # Process attention temporally
        x_TC = rearrange(x_SC, '(B T) S C -> (B S) T C', T=T)
        temporal_shift, temporal_scale, temporal_gate = repeat(self.temporal_adaLN(cond), 'B C -> (B S) 1 C', S=S).chunk(3, dim=-1)
        x_TC = x_TC + temporal_gate * self.temporal_attn(modulate(x_TC, temporal_shift, temporal_scale), causal=True)

        # Apply the MLP
        mlp_shift, mlp_scale, mlp_gate = repeat(self.mlp_adaLN(cond), 'B C -> (B S) 1 C', S=S).chunk(3, dim=-1)
        x_TC = x_TC + mlp_gate * self.mlp(modulate(self.norm2(x_TC), mlp_shift, mlp_scale))
        x_TSC = rearrange(x_TC, '(B S) T C -> B T S C', S=S)
        return x_TSC


class STTransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        qk_norm: bool = True,
        use_mup: bool = True,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
        mlp_bias: bool = True,
        mlp_drop: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([STBlock(
            num_heads=num_heads,
            d_model=d_model,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            qk_norm=qk_norm,
            use_mup=use_mup,
            attn_drop=attn_drop,
            mlp_ratio=mlp_ratio,
            mlp_bias=mlp_bias,
            mlp_drop=mlp_drop,
        ) for _ in range(num_layers)])

    def forward(self, tgt, cond):
        x = tgt
        for layer in self.layers:
            x = layer(x, cond)

        return x
