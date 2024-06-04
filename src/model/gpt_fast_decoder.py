from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_activation_fn


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def map_state_dict(state_dict):
    # map original state key to gpt_fast key
    new_state = {}
    for k, v in state_dict.items():
        if "decoder" not in k:
            new_state[k] = v
            continue
        # decoder.decoder.layers.0.self_attn.in_proj_weight -> decoder.layers.0.self_attn.in_proj_weight
        k = k[8:]  # remove "decoder." prefix

        # map self_attn state
        if "self_attn" in k:
            # decoder.layers.0.self_attn.in_proj_weight -> decoder.layers.0.self_attn.wqkv.weight
            k = k.replace("in_proj_weight", "wqkv.weight")
            k = k.replace("in_proj_bias", "wqkv.bias")
            k = k.replace("out_proj.weight", "wo.weight")
            k = k.replace("out_proj.bias", "wo.bias")

        # map multihead_attn state
        if "multihead_attn.in_proj" in k:
            # split weight to q, k, v
            # decoder.layers.0.multihead_attn.in_proj_weight -> decoder.layers.0.multihead_attn.query.weight, decoder.layers.0.multihead_attn.key.weight, decoder.layers.0.multihead_attn.value.weight
            part = "weight" if "weight" in k else "bias"
            prefix = k[: k.find(f"in_proj_{part}")]
            assert v.shape[0] % 3 == 0
            split_dim = v.shape[0] // 3
            q_weight, k_weight, v_weight = v.split(
                [split_dim, split_dim, split_dim], dim=0
            )
            new_state[prefix + f"query.{part}"] = q_weight
            new_state[prefix + f"key.{part}"] = k_weight
            new_state[prefix + f"value.{part}"] = v_weight
            continue

        if "multihead_attn.out_proj" in k:
            # decoder.layers.0.multihead_attn.out_proj.weight -> decoder.layers.0.multihead_attn.out.weight
            k = k.replace("out_proj.weight", "out.weight")
            k = k.replace("out_proj.bias", "out.bias")

        new_state[k] = v
    return new_state


@dataclass
class ModelArgs:
    n_layer: int = 4
    n_head: int = 12
    dim: int = 768
    intermediate_size: int = None
    head_dim: int = 64

    def __post_init__(self):
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head


class KVCache(nn.Module):
    def __init__(
        self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16
    ):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class GPTFastDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        nlayer: int,
        ff_ratio: int = 4,
    ) -> None:
        super().__init__()

        config = ModelArgs(
            n_layer=nlayer,
            n_head=nhead,
            dim=d_model,
            intermediate_size=d_model * ff_ratio,
        )
        self.config = config

        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layer)
        )

        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        for b in self.layers:
            b.multihead_attn.k_cache = None
            b.multihead_attn.v_cache = None

        if (
            self.max_seq_length >= max_seq_length
            and self.max_batch_size >= max_batch_size
        ):
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size

        for b in self.layers:
            b.self_attn.kv_cache = KVCache(
                max_batch_size,
                max_seq_length,
                self.config.n_head,
                head_dim,
                dtype,
            )
            b.multihead_attn.k_cache = None
            b.multihead_attn.v_cache = None

        self.causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        input_pos: Tensor,
    ) -> Tensor:
        if self.training:
            raise ValueError("GPTFastDecoder only supports inference.")

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            # https://github.com/pytorch-labs/gpt-fast/issues/31
            output = x
            tgt_mask = self.causal_mask[None, None, input_pos]
            for i, layer in enumerate(self.layers):
                output = layer(output, memory, input_pos=input_pos, tgt_mask=tgt_mask)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.multihead_attn = CrossAttention(config)

        layer_norm_eps = 1e-5

        d_model = config.dim
        dim_feedforward = config.intermediate_size

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = True
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn("gelu")

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        input_pos: Tensor,
    ) -> Tensor:
        assert self.norm_first is True
        x = tgt
        x = x + self.self_attn(self.norm1(x), tgt_mask, input_pos)
        x = x + self.multihead_attn(self.norm2(x), memory)
        x = x + self._ff_block(self.norm3(x))
        return x

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, 3 * config.dim)
        self.wo = nn.Linear(config.dim, config.dim)

        self.kv_cache: Optional[KVCache] = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.dim = config.dim

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        input_pos: Optional[Tensor] = None,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_head * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        self.query = nn.Linear(config.dim, config.dim)
        self.key = nn.Linear(config.dim, config.dim)
        self.value = nn.Linear(config.dim, config.dim)
        self.out = nn.Linear(config.dim, config.dim)

        self.k_cache = None
        self.v_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim

    def get_kv(self, xa: torch.Tensor):
        if self.k_cache is not None and self.v_cache is not None:
            return self.k_cache, self.v_cache

        k = self.key(xa)
        v = self.value(xa)

        # Reshape for correct format
        batch_size, source_seq_len, _ = k.shape
        k = k.view(batch_size, source_seq_len, self.n_head, self.head_dim)
        v = v.view(batch_size, source_seq_len, self.n_head, self.head_dim)

        if self.k_cache is None:
            self.k_cache = k
        if self.v_cache is None:
            self.v_cache = v

        return k, v

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
    ):
        q = self.query(x)
        batch_size, target_seq_len, _ = q.shape
        q = q.view(batch_size, target_seq_len, self.n_head, self.head_dim)
        k, v = self.get_kv(xa)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        wv = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=False,
        )
        wv = wv.transpose(1, 2).reshape(
            batch_size,
            target_seq_len,
            self.n_head * self.head_dim,
        )

        return self.out(wv)
