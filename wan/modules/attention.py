# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import os
import torch.distributed as dist

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'attention',
]


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.float16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


class _torch_nn_functional_scaled_dot_product_attention_MIGraphX(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
        q, k, v,
        attn_mask = None,
        is_causal = False,
        dropout_p = 0.0
    ):
        output = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=dropout_p
        )
        return output.transpose(1, 2)

def torch_nn_functional_scaled_dot_product_attention_MIGraphX(hash_key, mgx_export_path = None):
    options = {"deallocate": True}
    if mgx_export_path != "":
        if dist.is_initialized():
            mgx_path = os.path.join(mgx_export_path, f"wan_attention.{hash_key}.rank_{dist.get_rank()}.mgx")
        else:
            mgx_path = os.path.join(mgx_export_path, f"wan_attention.{hash_key}.mgx")

        if os.path.exists(mgx_path):
            print(f"Load mgx model from : {mgx_path}")
            options.update({ "load_compiled": mgx_path })
        else:
            print(f"Export mgx model to : {mgx_path}")
            options.update({ "save_compiled": mgx_path })

    import torch_migraphx
    m = _torch_nn_functional_scaled_dot_product_attention_MIGraphX()
    m = torch.compile(m, backend='migraphx', options=options, dynamic=False)

    return m


class TorchAttnImplPool:
    def __init__(self):
        self.cache = {}
        self.key_type = os.environ["YUNCHANG_FLASH_ATTENTION_IMPL"] if "YUNCHANG_FLASH_ATTENTION_IMPL" in os.environ else "default"
        self.mgx_export_path = os.environ["YUNCHANG_MIGRAPHX_MODEL_EXPORT_DIR"] if "YUNCHANG_MIGRAPHX_MODEL_EXPORT_DIR" in os.environ else ""
        self.enable_finite_check = os.environ.get("YUNCHANG_ENABLE_FINITE_CHECK", "OFF") == "ON"
        print(f"Using TorchAttnImplPool as : {self.key_type}, Finite check : {self.enable_finite_check}")

    def use_specific_attention_impl(self):
        return self.key_type != "default"

    def compute_attention(
        self, q, k, v,
        attn_mask=None,
        is_causal=False,
        dropout_p=0.0
    ):
        key = (
            q.shape, q.dtype, q.device,
            k.shape, k.dtype,
            v.shape, v.dtype,
            attn_mask,
            is_causal,
            dropout_p
        )
        hash_key = hash(key)

        if key not in self.cache:
            if self.key_type == "torch.nn.functional.scaled_dot_product_attention+MIGraphX":
                self.cache[key] = torch_nn_functional_scaled_dot_product_attention_MIGraphX(hash_key, self.mgx_export_path)
            else:
                raise ValueError(f"Unsupported key type: {self.key_type}")

        m = self.cache[key]

        output = m(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=dropout_p
        )

        if self.enable_finite_check:
            if torch.isfinite(output).all().item():
                return output
            else:
                return None
        else:
            return output

attn_pool = TorchAttnImplPool()

def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.float16,
    fa_version=None,
):
    # -----------------------------------------------------------------------------
    # Use the TorchAttnImplPool to compute attention
    if attn_pool.use_specific_attention_impl():
        out = attn_pool.compute_attention(
            q, k, v,
            attn_mask=None, is_causal=causal, dropout_p=dropout_p
        )

        # out will be None if the original value contains a non-finite
        # value when ENABLE_FINITE_CHECK is enabled.
        # In this case, we will fall back to the original flash_attn path.
        if out is not None:
            return out
    

    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
        )
    attn_mask = None

    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

    out = out.transpose(1, 2).contiguous()
    return out
