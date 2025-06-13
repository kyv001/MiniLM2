import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import ( # type: ignore
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
    PreTrainedTokenizerFast,
    AutoTokenizer
)
from transformers.modeling_outputs import CausalLMOutputWithPast # type: ignore

class MiniLM2Tokenizer(PreTrainedTokenizerFast):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)
    
    def _decode(self, token_ids, **kwargs):
        return self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))

class RotatoryPositionalEncoding(nn.Module):
    """旋转位置编码"""

    def __init__(self, dim: int, max_length: int):
        super().__init__()
        assert dim % 2 == 0
        positions = torch.arange(0, max_length, 1)
        theta = 1 / 10000 ** (torch.arange(0, dim, 2) / dim)  # thetai = 1/10000^(2i/dim)
        """
            theta0  theta1  theta2  theta3 ... theta(dim/2-1)
        m=0 0theta0 0theta1 0theta2 0theta3
        m=1 1theta0 1theta1 1theta2 1theta3
        m=2 2theta0 2theta1 2theta2 2theta3
        m=3 3theta0 3theta1 3theta2 3theta3
        ...
        m=max_length-1                         ...
        """
        positions_theta = positions.unsqueeze(1) * theta.unsqueeze(0)  # (max_length, dim//2)
        positions_sin = torch.sin(positions_theta)
        positions_cos = torch.cos(positions_theta)
        self.register_buffer('positions_sin', positions_sin)
        self.register_buffer('positions_cos', positions_cos)
        self.dim = dim

    def forward(self, x: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
        x_real = x[..., :self.dim // 2]  # (x.size(-2), dim//2)
        x_imag = x[..., self.dim // 2:]
        pos_cos = self.positions_cos[offset:offset + x.size(-2)] # type: ignore # (x.size(-2), dim//2)
        pos_sin = self.positions_sin[offset:offset + x.size(-2)] # type: ignore
        y_real = x_real * pos_cos - x_imag * pos_sin
        y_imag = x_real * pos_sin + x_imag * pos_cos
        return torch.cat([y_real, y_imag], dim=-1)

class GPTConfig(PretrainedConfig):
    model_type = "gpt"
    # 默认参数
    vocab_size: int = 32768
    dim = 768
    n_blocks = 12
    n_heads = 12
    max_position_embeddings = 1024
    dropout = .0

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.dim = dim
        self.u_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.u_proj(x)
        v = self.v_proj(x)
        return self.o_proj(self.dropout(u * nn.functional.silu(v)))

class SelfAttention(nn.Module):
    """带因果关系的多头自注意力，使用Flash Attention和RoPE"""
    def __init__(self, dim: int, max_length: int, n_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads==0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.pe = RotatoryPositionalEncoding(self.head_dim, max_length)
        self.dropout = dropout
        self.max_length = max_length

    def forward(self, x: torch.Tensor, *, past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None):
        B, T, C = x.shape

        # (B, T, C) -proj-> (B, T, C)
        # -view-> (B, T, n_heads, head_dim)
        # -T(1, 2)-> (B, n_heads, T, head_dim)
        if past_key_values is not None:
            k_cache, v_cache = past_key_values
            cache_size = k_cache.size(-2)
            xx = x[..., cache_size:, :]
            TT = T - cache_size
            k = self.k_norm(self.k_proj(xx).view(B, TT, self.n_heads, -1)).transpose(1, 2)
            v = self.v_proj(xx).view(B, TT, self.n_heads, -1).transpose(1, 2)
            k = torch.cat([k_cache, k], dim=-2)
            v = torch.cat([v_cache, v], dim=-2)
            q = self.q_norm(self.q_proj(xx).view(B, TT, self.n_heads, -1)).transpose(1, 2)
        else:
            cache_size = 0
            TT = T
            k = self.k_norm(self.k_proj(x).view(B, T, self.n_heads, -1)).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_heads, -1).transpose(1, 2)
            q = self.q_norm(self.q_proj(x).view(B, T, self.n_heads, -1)).transpose(1, 2)
        
        k_cache, v_cache = k.clone().detach(), v.clone().detach()

        q = self.pe(q, offset=cache_size).to(x.dtype)
        k = self.pe(k).to(x.dtype)

        # (B, n_heads, T, head_dim) -T(1, 2)-> (B, T, n_heads, head_dim)
        # -view-> (B, T, C)
        x = (
            nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout
            )
            .transpose(1, 2)
            .reshape(B, TT, C)
        )

        return self.o_proj(x), (k_cache, v_cache)

class GPTBlock(nn.Module):
    """一个Decoder块"""
    def __init__(self, dim: int, max_length: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = SelfAttention(dim, max_length, n_heads, dropout)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = MLP(dim, dim * 4, dropout)

    def forward(self, x: torch.Tensor, past_key_values: tuple[torch.Tensor, torch.Tensor] | None = None):
        attn_out, (k, v) = self.attn(self.norm1(x), past_key_values=past_key_values)
        xx = x[..., -attn_out.size(-2):, :]
        xx = xx + attn_out
        xx = xx + self.mlp(self.norm2(xx))
        return F.pad(xx, (0, 0, x.size(-2) - xx.size(-2), 0)), (k, v)

class GPT(PreTrainedModel, GenerationMixin):
    """大模型本体"""
    config_class = GPTConfig
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.wte = nn.Embedding(self.config.vocab_size, self.config.dim)
        self.blocks = nn.ModuleList([
            GPTBlock(
                self.config.dim,
                self.config.max_position_embeddings,
                self.config.n_heads,
                self.config.dropout
            ) for _ in range(self.config.n_blocks)
        ])
        self.lm_head = nn.Linear(self.config.dim, self.config.vocab_size)
        self.apply(self.init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * self.config.n_blocks) ** 0.5)

    def forward(self, x: torch.Tensor,
            return_dict: bool = False,
            past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
            use_cache=False):
        B, T = x.shape
        x = self.wte(x)
        key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i in range(self.config.n_blocks):
            block = self.blocks[i]
            if past_key_values is not None:
                x, (k, v) = block(x, past_key_values=past_key_values[i])
            else:
                x, (k, v) = block(x)
            key_values.append((k, v))
        x = self.lm_head(x)
        if not return_dict:
            return x
        return CausalLMOutputWithPast(logits=x, past_key_values=tuple(key_values))

    def prepare_inputs_for_generation(self, input_ids,
            past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
            use_cache=False,
            token_type_ids=None,
            attention_mask=None,
            **kwargs):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0) # 如果缺少batch维度，手动加上
        if input_ids.size(-1) > self.config.max_position_embeddings: # 如果超出了最大长度，将前面多出的部分截断
            cut_idx = input_ids.size(-1) - self.config.max_position_embeddings
            input_ids = input_ids[..., cut_idx:]
            if past_key_values is not None:
                past_key_values = tuple(
                    (k[..., 1:, :], v[..., 1:, :]) for k, v in past_key_values
                ) # 我们假定每次生成一个token，所以只需要去掉第一个位置
            
        return {"x": input_ids, "use_cache": use_cache, "past_key_values": past_key_values if use_cache else None}

    def init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

AutoConfig.register("gpt", GPTConfig)
AutoModelForCausalLM.register(GPTConfig, GPT)
AutoTokenizer.register(GPTConfig, fast_tokenizer_class=MiniLM2Tokenizer)
