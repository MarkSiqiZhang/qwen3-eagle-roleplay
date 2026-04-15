"""
EAGLE 3 Draft Model Architecture for Training.

Extracted and simplified from EAGLE/eagle/traineagle3/cnets.py with fixes
for Qwen3 compatibility (head_dim mismatch).
"""

import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


@torch.no_grad()
def padding(tensor, left=True):
    """Shift tensor left or right by 1 position, padding with zeros."""
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        return torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        return torch.cat((tensor[:, 1:], zeropadding), dim=1)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        # Fix for Qwen3: use config.head_dim if available
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(self.hidden_size * 2, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def _init_rope(self):
        rope_theta = getattr(self.config, "rope_theta", 10000)
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings,
                base=rope_theta,
            )
        else:
            raise NotImplementedError("Scaled RoPE not implemented for training")

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        lck = len(cache_hidden[0])

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids + lck)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Avoid in-place modification of cache (needed for gradient checkpointing)
        local_cache_k = list(cache_hidden[0]) if cache_hidden is not None else []
        local_cache_v = list(cache_hidden[1]) if cache_hidden is not None else []
        local_cache_k.append(key_states)
        local_cache_v.append(value_states)

        k0 = local_cache_k[0]
        v0 = local_cache_v[0]

        attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(self.head_dim)
        attn_weights = attn_weights + attention_mask

        for i in range(1, len(local_cache_k)):
            ki = local_cache_k[i]
            attn_weightsi = (query_states * ki).sum(-1) / math.sqrt(self.head_dim)
            attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights0 = attn_weights[..., :q_len]
        attn_output = torch.matmul(attn_weights0, v0)

        for i in range(1, len(local_cache_k)):
            vi = local_cache_v[i]
            attn_weightsi = attn_weights[..., q_len + i - 1]
            attn_outputi = attn_weightsi[..., None] * vi
            attn_output = attn_output + attn_outputi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, [local_cache_k, local_cache_v]


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayeremb(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: List[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)

        hidden_states, latest_hidden_cache = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states,), latest_hidden_cache


class Eagle3DraftModel(nn.Module):
    """EAGLE 3 draft model for training."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.fc = nn.Linear(config.hidden_size * 3, config.hidden_size, bias=False)
        self.midlayer = LlamaDecoderLayeremb(config)
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)

        self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))

        self.gradient_checkpointing = False

    @classmethod
    def from_pretrained(cls, draft_model_path, target_model, config=None):
        """Load pretrained draft model weights and target model embeddings."""
        from safetensors.torch import load_file
        import json

        if config is None:
            from transformers import PretrainedConfig
            config = PretrainedConfig.from_pretrained(draft_model_path)

        model = cls(config)

        # Load draft model weights
        safetensors_path = os.path.join(draft_model_path, "model.safetensors")
        state_dict = load_file(safetensors_path)
        model.load_state_dict(state_dict, strict=False)

        # Load embeddings from target model (frozen)
        model.embed_tokens.weight.data = target_model.model.embed_tokens.weight.data.clone()
        model.embed_tokens.requires_grad_(False)

        return model

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, dtype, device, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, dtype, device=device, past_key_values_length=past_key_values_length
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, dtype, tgt_len=input_shape[-1]).to(device)
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )
        return combined_attention_mask

    def train_forward(self, hidden_states_cat, target_logits, input_ids, attention_mask,
                      loss_mask, num_steps=7, loss_decay=0.8):
        """
        Training forward pass with multi-step autoregressive prediction.

        Args:
            hidden_states_cat: Concatenated hidden states from target [B, L, 3*H]
            target_logits: Target model logits [B, L, V] (already shifted left by 1)
            input_ids: Input token ids [B, L] (already shifted left by 1)
            attention_mask: [B, L]
            loss_mask: [B, L, 1] (1 for assistant tokens)
            num_steps: Number of prediction steps
            loss_decay: Decay factor for loss weighting
        """
        batch_size, seq_length, _ = hidden_states_cat.shape

        if self.gradient_checkpointing and self.training and not hidden_states_cat.requires_grad:
            hidden_states_cat.requires_grad = True

        hidden_states = self.fc(hidden_states_cat)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        attn_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length),
            hidden_states.dtype, hidden_states.device, 0
        )

        losses = []
        accuracies = []
        cache_hidden = [[], []]

        for idx in range(num_steps):
            last = idx == num_steps - 1
            inputs_embeds = self.embed_tokens(input_ids)
            if self.gradient_checkpointing and self.training and not inputs_embeds.requires_grad:
                inputs_embeds.requires_grad = True
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, None, False)
                    return custom_forward

                layer_outputs, cache_hidden = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.midlayer),
                    inputs_embeds, hidden_states, cache_hidden,
                    attn_mask, position_ids,
                )
            else:
                layer_outputs, cache_hidden = self.midlayer(
                    input_emb=inputs_embeds,
                    hidden_states=hidden_states,
                    cache_hidden=cache_hidden,
                    attention_mask=attn_mask,
                    position_ids=position_ids,
                    use_cache=True,
                )

            hidden_states_out = layer_outputs[0]

            # Compute loss
            with torch.no_grad():
                target_head = target_logits
                target_max_token = target_head.argmax(-1)
                self.t2d = self.t2d.to(target_max_token.device)
                target_mask = self.t2d[target_max_token]
                target_mask = target_mask[..., None].int()
                position_mask = target_mask * loss_mask
                target_head_draft = target_head[..., self.t2d]
                target_head_draft = target_head_draft.float()
                target_p = nn.Softmax(dim=2)(target_head_draft)
                target_p = target_p.detach()

            hidden_states = hidden_states_out
            hidden_states_normed = self.norm(hidden_states_out)
            logits = self.lm_head(hidden_states_normed).float()
            out_logp = nn.LogSoftmax(dim=2)(logits)
            plogp = target_p * out_logp
            loss = -torch.sum(position_mask * plogp, 2).mean()
            losses.append(loss)

            with torch.no_grad():
                acc = ((logits.argmax(-1) == target_p.argmax(-1)) * position_mask.squeeze(-1)).sum().item()
                acc = acc / (loss_mask.sum().item() + 1e-6)
                accuracies.append(acc)

            if not last:
                input_ids = padding(input_ids, left=False)
                target_logits = padding(target_logits, left=False)
                loss_mask = padding(loss_mask, left=False)

        # Weighted loss
        total_loss = sum(loss_decay ** i * losses[i] for i in range(len(losses)))
        avg_acc = sum(accuracies) / len(accuracies)

        return total_loss, avg_acc, losses, accuracies


# Attention mask utilities (from transformers)
def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask, dtype, tgt_len=None):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
