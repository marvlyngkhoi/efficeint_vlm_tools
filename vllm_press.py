# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv

# Original KV Press
# from transformers import (
#     LlamaForCausalLM,
#     MistralForCausalLM,
#     Phi3ForCausalLM,  
#     PreTrainedModel,
#     QuantizedCache,
#     Qwen2ForCausalLM,
# )
from transformers import(
    Idefics3ForConditionalGeneration,
    PreTrainedModel,
    QuantizedCache,
    Qwen2VLForConditionalGeneration
)

logger = logging.getLogger(__name__)








@dataclass
class BasePress:
    """
    Base class for all KV cache compression methods.
    The `forward_hook` method is called after the forward pass of an attention layer to update the cache.
    """

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The core logic of the compression method.

        Parameters
        ----------
        module :
            Transformer layer, see `hook` method for more details
        hidden_states :
            Hidden states of the layer
        keys :
            Keys of the cache (unquantized)
        values :
            Values of the cache (unquantized)
        attentions :
            Attention weights of the layer
        kwargs :
            Keyword arguments, as given to the forward pass of the layer

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Updated keys and values
        """

        raise NotImplementedError("compress method must be implemented in subclass")

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """
        Default forward hook called after the forward pass of an attention layer.
        The hook calls the compress method to compress the KV cache while ensuring:
            - compression is only applied only during the pre-filling phase
            - KV cache quantization is handled correctly

        Parameters
        ----------
        module :
            Transformer attention layer.
        input :
            Input to the hook. This is the input to the forward pass of the layer.
        kwargs :
            Keyword arguments, as given to the forward pass of the layer.
        output :
            Output of the hook. This is the original output of the forward pass of the layer.

        Returns
        -------
            Modified output of the forward pass of the layer.

        """
        
        # See e.g. LlamaDecoderLayer.forward for the output structure
        if len(output) == 3:
            _, attentions, cache = output
        else:
            attentions, cache = None, output[-1]

        hidden_states = kwargs["hidden_states"]
        q_len = hidden_states.shape[1]

        # Don't compress after pre-filling
        if cache.seen_tokens > q_len:
            return output

        if isinstance(cache, QuantizedCache):
            keys = cache._dequantize(cache._quantized_key_cache[module.layer_idx])
            values = cache._dequantize(cache._quantized_value_cache[module.layer_idx])
        else:
            keys = cache.key_cache[module.layer_idx]
            values = cache.value_cache[module.layer_idx]

        keys, values = self.compress(module, hidden_states, keys, values, attentions, kwargs)

        if isinstance(cache, QuantizedCache):
            cache._quantized_key_cache[module.layer_idx] = cache._quantize(keys, axis=cache.axis_key)
            cache._quantized_value_cache[module.layer_idx] = cache._quantize(values, axis=cache.axis_value)
            cache.key_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache.value_cache[module.layer_idx] = torch.zeros(0, dtype=keys.dtype, device=keys.device)
            cache._seen_tokens = keys.shape[2]
        else:
            cache.key_cache[module.layer_idx] = keys
            cache.value_cache[module.layer_idx] = values

        return output

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Context manager to apply a compression method to a model.
        Apply this context manager during the pre-filling phase to compress the context.

        Parameters
        ----------
        model : PreTrainedModel
            Model to apply the compression method to
        """

        # if not isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Phi3ForCausalLM, Qwen2ForCausalLM)):
        #     logger.warning(f"Model {type(model)} not tested")

        hooks = []
        # original KV press
        # if  isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Phi3ForCausalLM, Qwen2ForCausalLM)):
        #     try:
        #         for layer in model.model.layers:
        #             hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
    
        #         yield
        #     finally:
        #         for forward_hook in hooks:
        #             forward_hook.remove()

        # VLM here we are testing for smolvlm which is based on Idefics3 model
        if  isinstance(model, (Idefics3ForConditionalGeneration)):
            try:
                for layer in model.model.text_model.layers:
                    hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
    
                yield
            finally:
                for forward_hook in hooks:
                    forward_hook.remove()
        elif  isinstance(model, (Qwen2VLForConditionalGeneration)):
            try:
                for layer in model.model.layers:
                    hooks.append(layer.self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))
    
                yield
            finally:
                for forward_hook in hooks:
                    forward_hook.remove()
        



@dataclass
class ScorerPress(BasePress):
    """
    Default press method for using a score method.
    Any ScorerPress subclass must implement the `score` method that computes a tensor of scores for each key-value pair
    The KV pairs with the lowest scores will be pruned in the `compress` method.
    The cache is uniformly pruned across all heads and layers using the compression_ratio parameter.
    """

    compression_ratio: float = 0.0

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute a tensor of scores with shape (bsz, num_key_value_heads, q_len)
        The KV pairs with lowest scores will be pruned in the `compress` method.
        """
        raise NotImplementedError

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.compression_ratio == 0:
            return keys, values

        # Compute scores
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Get indices of KV pairs with the lowest scores
        q_len = hidden_states.shape[1]
        n_kept = int(q_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values


@dataclass
class KnormPress(ScorerPress):
    """Prune KV pairs with highest L2 norm of keys (https://arxiv.org/pdf/2406.11430)"""

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        return -keys.norm(dim=-1)
 
      
@dataclass
class RandomPress(ScorerPress):
    """Randomly prune KV pairs"""

    compression_ratio: float = 0.0
    seed: Optional[int] = None

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        if self.seed is not None:
            torch.manual_seed(self.seed)
        return torch.rand(*keys.shape[:-1]).to(keys.device, keys.dtype)
        
        
@dataclass
class ExpectedAttentionPress(ScorerPress):
    """
    Compute scores based on the expected attention on next positions. To do so
        1. Compute the mean and covariance matrix of the queries before RoPE.
        2. Compute the RoPE rotation matrix R on next n_future_positions and average it
        3. Apply R to the mean and covariance matrice of the queries.
        4. As attention A = exp(Q @ K / sqrt(d)), we compute the expected attention
        E(A) = exp(K @ mean.T / sqrt(d) + 1/2 K @ cov @ K.T / d)
        5. Rescale the scores by the norm of the values
    The first n_sink tokens are removed from calculations (sink attention phenomenon).
    """

    compression_ratio: float = 0.0
    n_future_positions: int = 512
    n_sink: int = 4
    use_covariance: bool = True
    use_vnorm: bool = True

    def get_query_statistics(self, module: nn.Module, hidden_states: torch.Tensor):
        """
        Compute the mean and covariance matrix of the queries
        """

        bsz, q_len, _ = hidden_states.shape
        n, d = module.config.num_attention_heads, module.head_dim

        # Remove first hidden_states that likely contain outliers
        h = hidden_states[:, self.n_sink :]

        if hasattr(module, "q_proj"):
            Wq = module.q_proj.weight
        elif hasattr(module, "qkv_proj"):
            Wq = module.qkv_proj.weight[: n * d]
        else:
            raise NotImplementedError(f"ExpectedAttentionPress not yet implemented for {module.__class__}.")

        # Query mean
        mean_h = torch.mean(h, dim=1, keepdim=True)
        mu = torch.matmul(mean_h, Wq.T).squeeze(1)
        mu = mu.view(bsz, n, d)

        # Query covariance
        cov = None
        if self.use_covariance:
            h = h - mean_h
            cov = torch.matmul(h.transpose(1, 2), h) / h.shape[1]
            cov = torch.matmul(Wq, torch.matmul(cov, Wq.T))  # TODO: not optimal
            cov = cov.view(bsz, n, d, n, d).diagonal(dim1=1, dim2=3)
            cov = cov.permute(0, 3, 1, 2)

        # RoPE rotation matrix on next n_future_positions
        position_ids = torch.arange(q_len, q_len + self.n_future_positions).unsqueeze(0).to(mu.device)
        cos, sin = module.rotary_emb(mu, position_ids)
        cos, sin = cos[0], sin[0]

        Id = torch.eye(d, device=cos.device, dtype=cos.dtype)
        P = torch.zeros((d, d), device=cos.device, dtype=cos.dtype)
        P[d // 2 :, : d // 2], P[: d // 2, d // 2 :] = torch.eye(d // 2), -torch.eye(d // 2)
        R = cos.unsqueeze(1) * Id + sin.unsqueeze(1) * P

        # Apply average rotation to the mean and covariance
        R = R.mean(dim=0)
        mu = torch.matmul(mu, R.T)
        if self.use_covariance:
            cov = torch.matmul(R, torch.matmul(cov, R.T))

        # Instead of using the average rotation matrix, we could use a mixture of gaussian statistics to
        # estimate mean and covariance. Estimation is better, but end-to-end performance was lower.
        # mu = torch.einsum("bhj, fij -> bhfi", mu, R)
        # mean_mu = mu.mean(dim=2, keepdim=True)
        # if self.use_covariance:
        #     cov = torch.einsum("fki, bhkl, fjl -> bhfij", R, cov, R)
        #     cov = cov.mean(dim=2)
        #     cov += torch.einsum("bhfi, bhfj -> bhji", mu - mean_mu, mu - mean_mu) / self.n_future_positions
        # mu = mean_mu.squeeze(2)

        return mu, cov

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        # Remove sink tokens
        assert keys.size(2) > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"
        keys = keys[:, :, self.n_sink :]
        values = values[:, :, self.n_sink :]

        # Compute query statistics
        mean_query, cov_query = self.get_query_statistics(module, hidden_states)

        # Compute scores
        bsz, num_key_value_heads, q_len, d = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        keys = repeat_kv(keys, num_key_value_groups).transpose(2, 3)
        scores = torch.matmul(mean_query.unsqueeze(2), keys).squeeze(2) / math.sqrt(d)
        if self.use_covariance:
            scores += torch.einsum("bhin, bhij, bhjn->bhn", keys, cov_query, keys) / d / 2
        scores = F.softmax(scores, dim=-1)

        # Average scores across groups
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len)
        scores = scores.mean(dim=2)

        # Rescale scores by the norm of the values
        if self.use_vnorm:
            scores = scores * values.norm(dim=-1)

        # Add back the sink tokens. Use max score to make sure they are not pruned.
        scores = F.pad(scores, (self.n_sink, 0), value=scores.max().item())

        return scores
