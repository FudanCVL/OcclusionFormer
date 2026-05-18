


from typing import Any, Dict, Optional, Tuple, Union, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    FluxAttnProcessor2_0,
    FluxAttnProcessor2_0_NPU,
    FusedFluxAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed, CombinedTimestepLabelEmbeddings
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, FP32SiLU, SwiGLU
import torch.nn.functional as F

logger = logging.get_logger(__name__)

from .config import (
    LAYOUT_DUAL_STREAM_BLOCK_INDICES,
    LAYOUT_SINGLE_STREAM_BLOCK_INDICES,
    MASK_PREDICTOR_IN_CHANNELS,
    MASK_PREDICTOR_HIDDEN_CHANNEL_1,
    MASK_PREDICTOR_HIDDEN_CHANNEL_2,
    MASK_PREDICTOR_OUT_CHANNELS,
    MASK_PREDICTOR_KERNEL_SIZE,
    MASK_PREDICTOR_PADDING,
    LAYOUT_SIGMA_SCALE,
    LAYOUT_GUIDANCE_SCALE,
)
from ..transformer_utils import FeedForward


from .tools import (
    AdaLayerNormZeroLayout,
    FluxLayoutNestedAttnProcessor2_0,
    TextBoundingboxProjection,
    enable_lora,
    is_occlusion_mode,
    zero_module,
)


@dataclass
class Transformer2DModelOutput(BaseOutput):
    sample: torch.Tensor
    z_buffer_masks: Optional[List[torch.Tensor]] = None


@maybe_allow_in_graph
class OcclusionFormerFluxSingleTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 4.0, block_type="default"):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        if is_torch_npu_available():
            deprecation_message = (
                "Defaulting to FluxAttnProcessor2_0_NPU for NPU devices will be removed. Attention processors "
                "should be set explicitly using the `set_attn_processor` method."
            )
            deprecate("npu_processor", "0.34.0", deprecation_message)
            processor = FluxAttnProcessor2_0_NPU()
        else:
            processor = FluxAttnProcessor2_0()

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )
        
        self.block_type = block_type
        if is_occlusion_mode(self.block_type):
            self.layout_forward = zero_module(nn.Linear(dim, dim))
            self.norm1_layout = AdaLayerNormZeroLayout(dim)
            self.layout_nested_attn_processor = FluxLayoutNestedAttnProcessor2_0()
            
            self.mask_predictor = nn.Sequential(
                nn.Conv2d(
                    MASK_PREDICTOR_IN_CHANNELS,
                    MASK_PREDICTOR_HIDDEN_CHANNEL_1,
                    kernel_size=MASK_PREDICTOR_KERNEL_SIZE,
                    padding=MASK_PREDICTOR_PADDING,
                ),
                nn.GELU(),
                nn.Conv2d(
                    MASK_PREDICTOR_HIDDEN_CHANNEL_1,
                    MASK_PREDICTOR_HIDDEN_CHANNEL_2,
                    kernel_size=MASK_PREDICTOR_KERNEL_SIZE,
                    padding=MASK_PREDICTOR_PADDING,
                ),
                nn.GELU(),
                nn.Conv2d(MASK_PREDICTOR_HIDDEN_CHANNEL_2, MASK_PREDICTOR_OUT_CHANNELS, kernel_size=1),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        layout_hidden_states: Optional[torch.Tensor] = None,
        layout_masks: Optional[torch.Tensor] = None,
        img_idxs_list_list: Optional[List[List[torch.Tensor]]] = None,
        layout_rotary_emb: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        enable_layout: bool = True,
        layout_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        layout_temb_list: Optional[List[torch.Tensor]] = None,
        layout_temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        temb_for_layout = layout_temb if layout_temb is not None else temb
        hidden_states_prompt = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states_prompt
        norm_hidden_states, gate = self.norm(hidden_states_prompt, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hidden_states_prompt = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states_prompt = gate * self.proj_out(hidden_states_prompt)
        hidden_states_prompt = residual + hidden_states_prompt
        if hidden_states_prompt.dtype == torch.float16:
            hidden_states_prompt = hidden_states_prompt.clip(-65504, 65504)
        encoder_hidden_states, hidden_states = (
                hidden_states_prompt[:, : encoder_hidden_states.shape[1]],
                hidden_states_prompt[:, encoder_hidden_states.shape[1] :],
            )

        z_buffer_masks_2ch = None
        if is_occlusion_mode(self.block_type) and enable_layout:
            occlusion = None
            bbox_mask = None
            img_height = None
            img_width = None
            if layout_kwargs is not None:
                if not isinstance(layout_kwargs, list):
                    layout_kwargs = [layout_kwargs]
                for layout in layout_kwargs:
                    layout_args = layout.get("layout", {})
                    if "occlusion" in layout_args:
                        occlusion = layout_args["occlusion"]
                    if "bbox_mask" in layout_args:
                        bbox_mask = layout_args["bbox_mask"]
                    if "img_height" in layout_args:
                        img_height = layout_args["img_height"]
                    if "img_width" in layout_args:
                        img_width = layout_args["img_width"]
                    break
            
            residual = hidden_states
            bsz, max_objs = layout_hidden_states.shape[:2]
            img_len = hidden_states.shape[1]
            layout_hidden_add = torch.zeros_like(layout_hidden_states)
            hidden_states_add = torch.zeros_like(hidden_states)
            img_add_cnt = torch.zeros((bsz, img_len)).to(dtype=hidden_states_add.dtype, device=hidden_states_add.device)
            img_weight_sum = torch.zeros((bsz, img_len, hidden_states.shape[-1])).to(dtype=hidden_states_add.dtype, device=hidden_states_add.device)

            valid_mask = (layout_masks == 1)
            valid_indices = valid_mask.nonzero(as_tuple=False)  
            if valid_indices.size(0) > 0:
                valid_bs = valid_indices[:, 0]
                valid_objs = valid_indices[:, 1]
                num_valid = valid_indices.size(0)
                
                layout_sigma_list = []
                layout_cosinesim_query_list = []
                
                for k in range(num_valid):
                    i = valid_bs[k].item()
                    j = valid_objs[k].item()
                    layout_sigma_item, layout_cosinesim_query_item = self.norm1_layout(
                        emb=layout_temb_list[i][0, j:j+1]
                    )
                    layout_sigma_list.append(layout_sigma_item)
                    layout_cosinesim_query_list.append(layout_cosinesim_query_item)
                
                layout_sigma = LAYOUT_SIGMA_SCALE * F.softplus(torch.cat(layout_sigma_list, dim=0)).to(layout_sigma_list[0].dtype)
                layout_cosinesim_query = torch.cat(layout_cosinesim_query_list, dim=0)
                
                all_scaled_outputs = []
                all_layout_outputs = []
                all_img_indices = []
                all_img_idxs_list = []
                all_layout_seqs = []
                all_tembs = []

                for k in range(num_valid):
                    i = valid_bs[k].item()
                    j = valid_objs[k].item()
                    img_idxs = img_idxs_list_list[i][j].to(norm_hidden_states.device).to(torch.int64)
                    all_img_idxs_list.append(img_idxs)
                    all_layout_seqs.append(torch.cat([layout_hidden_states[i, j].unsqueeze(0), hidden_states[i, img_idxs]], dim=0))
                    all_tembs.append(temb_for_layout[i])

                norm_seq_list = []
                gate_list = []
                mlp_list = []
                with enable_lora((
                    self.norm.linear,
                    self.proj_mlp,
                    self.attn.to_q,
                    self.attn.to_k,
                    self.attn.to_v,
                    self.proj_out,
                ), on=True, off=False):
                    for seq, temb_item in zip(all_layout_seqs, all_tembs):
                        norm_seq, gate_item = self.norm(seq.unsqueeze(0), emb=temb_item.unsqueeze(0))
                        mlp_seq = self.act_mlp(self.proj_mlp(norm_seq))
                        norm_seq_list.append(norm_seq.squeeze(0))
                        gate_list.append(gate_item.squeeze(0))
                        mlp_list.append(mlp_seq.squeeze(0))

                    attn_seq_list = self.layout_nested_attn_processor(
                        self.attn,
                        hidden_states=norm_seq_list,
                        encoder_hidden_states=None,
                        image_rotary_emb_list=layout_rotary_emb,
                    )

                    for k, (attn_seq, mlp_seq, gate_item, img_idxs) in enumerate(zip(attn_seq_list, mlp_list, gate_list, all_img_idxs_list)):
                        hidden_states_layout = torch.cat([attn_seq, mlp_seq], dim=1).unsqueeze(0)
                        hidden_states_layout = gate_item.view(1, 1, -1) * self.proj_out(hidden_states_layout)

                        layout_output = hidden_states_layout[:, :1].squeeze(0, 1)
                        img_output = hidden_states_layout[:, 1:].squeeze(0)
                        scaled_output = self.layout_forward(img_output)
                        all_scaled_outputs.append(scaled_output)
                        all_layout_outputs.append(layout_output)

                        i = valid_bs[k].item()
                        batch_indices = torch.full_like(img_idxs, i)
                        all_img_indices.append(torch.stack([batch_indices, img_idxs], dim=1))
                
                attn_output_add_list = []
                dim = all_scaled_outputs[0].shape[-1]
                for k in range(num_valid):
                    full_attn_output = torch.zeros(img_len, dim, 
                                                   device=hidden_states.device, 
                                                   dtype=hidden_states.dtype)
                    img_idxs = all_img_idxs_list[k]
                    src_output = all_scaled_outputs[k].to(dtype=full_attn_output.dtype)
                    full_attn_output.scatter_add_(
                        dim=0,
                        index=img_idxs.unsqueeze(-1).expand(-1, dim),
                        src=src_output
                    )
                    attn_output_add_list.append(full_attn_output)
                
                if occlusion is not None and bbox_mask is not None and all_scaled_outputs:
                    attn_output_add_stacked = torch.stack(attn_output_add_list, dim=0)
                    layout_sigma_value = layout_sigma
                    alpha = 1 - torch.exp(-layout_sigma_value)
                    
                    bbox_mask_len = len(bbox_mask) if isinstance(bbox_mask, list) else bbox_mask.shape[0]
                    bbox_masks_list = []
                    for k in range(num_valid):
                        obj_idx = valid_objs[k].item()
                        if obj_idx < bbox_mask_len:
                            bbox_masks_list.append(bbox_mask[obj_idx])
                        else:
                            bbox_masks_list.append(torch.zeros_like(bbox_mask[0]))
                    bbox_masks_stacked = torch.stack(bbox_masks_list, dim=0)
                    
                    z_buffer_mask_list = []
                    img_h = img_height if img_height is not None else int(img_len ** 0.5)
                    img_w = img_width if img_width is not None else int(img_len ** 0.5)
                    
                    for layout_idx in range(num_valid):
                        img_attn = attn_output_add_stacked[layout_idx]
                        query = layout_cosinesim_query[layout_idx]
                        
                        img_attn_normalized = F.normalize(img_attn, p=2, dim=-1)
                        query_normalized = F.normalize(query, p=2, dim=-1)
                        cosine_sim = (img_attn_normalized * query_normalized).sum(dim=-1)
                        
                        bbox_mask_1d = bbox_masks_stacked[layout_idx].squeeze(-1)
                        cosine_sim = cosine_sim * bbox_mask_1d
                        
                        z_mask_2d = cosine_sim.reshape(1, 1, img_h, img_w)
                        
                        z_mask_2ch = self.mask_predictor(z_mask_2d)
                        z_mask_2ch_flat = z_mask_2ch.reshape(1, 2, img_h * img_w).permute(0, 2, 1)
                        z_buffer_mask_list.append(z_mask_2ch_flat)
                    
                    z_buffer_masks_inner = torch.cat(z_buffer_mask_list, dim=0)
                    z_buffer_masks_2ch = z_buffer_masks_inner
                    
                    transmittance_list = []
                    for layout_idx in range(num_valid):
                        j = valid_objs[layout_idx].item()
                        occluder_indices = occlusion[j] if j < len(occlusion) else []
                        
                        if len(occluder_indices) > 0:
                            occluder_valid_indices = []
                            for occ_idx in occluder_indices:
                                mask = (valid_objs == occ_idx)
                                if mask.any():
                                    occluder_valid_indices.append(mask.nonzero(as_tuple=True)[0].item())
                            
                            if len(occluder_valid_indices) > 0:
                                occluder_bbox_masks = bbox_masks_stacked[occluder_valid_indices]
                                occluder_sigmas = layout_sigma_value[occluder_valid_indices]
                                mask_foreground = occluder_bbox_masks
                                accumulated_opacity = (mask_foreground * occluder_sigmas.unsqueeze(1)).sum(dim=0, keepdim=True)
                            else:
                                accumulated_opacity = torch.zeros(1, img_len, dim, device=bbox_masks_stacked.device, dtype=bbox_masks_stacked.dtype)
                        else:
                            accumulated_opacity = torch.zeros(1, img_len, dim, device=bbox_masks_stacked.device, dtype=bbox_masks_stacked.dtype)
                        
                        transmittance = torch.exp(-accumulated_opacity)
                        transmittance_list.append(transmittance.squeeze(0))
                    
                    transmittances = torch.stack(transmittance_list, dim=0)
                    mask_final = bbox_masks_stacked
                    weights = transmittances * alpha.unsqueeze(1) * mask_final
                    weighted_attn_outputs = weights * attn_output_add_stacked
                    
                    for k in range(num_valid):
                        i = valid_bs[k].item()
                        hidden_states_add[i] = hidden_states_add[i] + weighted_attn_outputs[k]
                        img_weight_sum[i] = img_weight_sum[i] + weights[k]
                        img_idxs = all_img_idxs_list[k]
                        img_add_cnt[i, img_idxs] = img_add_cnt[i, img_idxs] + 1
                    
                    layout_outputs = torch.stack(all_layout_outputs, dim=0)
                    layout_hidden_add[valid_indices[:, 0], valid_indices[:, 1]] = layout_outputs
                    
                elif all_scaled_outputs:
                    scaled_outputs = torch.cat(all_scaled_outputs, dim=0)
                    img_indices = torch.cat(all_img_indices, dim=0)
                    layout_outputs = torch.stack(all_layout_outputs, dim=0)

                    scaled_outputs = scaled_outputs.to(hidden_states_add.dtype)

                    hidden_states_add.view(-1, hidden_states_add.size(-1)).scatter_add_(
                        dim=0,
                        index=(img_indices[:, 0] * img_len + img_indices[:, 1]).unsqueeze(-1).expand(-1, scaled_outputs.size(-1)),
                        src=scaled_outputs
                    )

                    img_add_cnt.view(-1).scatter_add_(
                        dim=0,
                        index=img_indices[:, 0] * img_len + img_indices[:, 1],
                        src=torch.ones(img_indices.size(0)).to(hidden_states_add)
                    )

                    layout_hidden_add[valid_indices[:, 0], valid_indices[:, 1]] = layout_outputs
            
            weight_sum_nonzero = (img_weight_sum.sum(dim=-1) > 0).float().unsqueeze(-1).to(hidden_states_add.dtype)
            img_weight_sum_safe = (img_weight_sum + (img_weight_sum == 0).float() * 1e-8).to(hidden_states_add.dtype)
            attn_output_weighted = hidden_states_add / img_weight_sum_safe
            
            img_add_cnt_safe = (img_add_cnt.unsqueeze(-1) + (img_add_cnt.unsqueeze(-1) == 0).float()).to(hidden_states_add.dtype)
            attn_output_simple = hidden_states_add / img_add_cnt_safe
            
            attn_output_combined = weight_sum_nonzero * attn_output_weighted + (1 - weight_sum_nonzero) * attn_output_simple
            hidden_states = residual + attn_output_combined
            layout_hidden_states = layout_hidden_states + layout_hidden_add

        return encoder_hidden_states, hidden_states, layout_hidden_states, z_buffer_masks_2ch

@maybe_allow_in_graph
class OcclusionFormerFluxTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6, block_type="default"
    ):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)
        self.norm1_context = AdaLayerNormZero(dim)

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=FluxAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.block_type = block_type
        if is_occlusion_mode(self.block_type):
            self.layout_forward = zero_module(nn.Linear(dim, dim))
            self.norm1_layout = AdaLayerNormZeroLayout(dim)
            self.layout_nested_attn_processor = FluxLayoutNestedAttnProcessor2_0()
            
            self.mask_predictor = nn.Sequential(
                nn.Conv2d(
                    MASK_PREDICTOR_IN_CHANNELS,
                    MASK_PREDICTOR_HIDDEN_CHANNEL_1,
                    kernel_size=MASK_PREDICTOR_KERNEL_SIZE,
                    padding=MASK_PREDICTOR_PADDING,
                ),
                nn.GELU(),
                nn.Conv2d(
                    MASK_PREDICTOR_HIDDEN_CHANNEL_1,
                    MASK_PREDICTOR_HIDDEN_CHANNEL_2,
                    kernel_size=MASK_PREDICTOR_KERNEL_SIZE,
                    padding=MASK_PREDICTOR_PADDING,
                ),
                nn.GELU(),
                nn.Conv2d(MASK_PREDICTOR_HIDDEN_CHANNEL_2, MASK_PREDICTOR_OUT_CHANNELS, kernel_size=1),
            )


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        layout_hidden_states: Optional[torch.Tensor] = None,
        layout_masks: Optional[torch.Tensor] = None,
        img_idxs_list_list: Optional[List[List[torch.Tensor]]] = None,
        layout_rotary_emb: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        enable_layout: bool = True,
        layout_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        layout_temb_list: Optional[List[torch.Tensor]] = None,
        layout_temb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
        joint_attention_kwargs = joint_attention_kwargs or {}
        
        occlusion = None
        bbox_mask = None
        img_height = None
        img_width = None
        if layout_kwargs is not None:
            if not isinstance(layout_kwargs, list):
                layout_kwargs = [layout_kwargs]
            for layout in layout_kwargs:
                layout_args = layout.get("layout", {})
                if "occlusion" in layout_args:
                    occlusion = layout_args["occlusion"]
                if "bbox_mask" in layout_args:
                    bbox_mask = layout_args["bbox_mask"]
                if "img_height" in layout_args:
                    img_height = layout_args["img_height"]
                if "img_width" in layout_args:
                    img_width = layout_args["img_width"]
                break

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output


        z_buffer_masks_2ch = None
        if is_occlusion_mode(self.block_type) and enable_layout:
            temb_for_layout = layout_temb if layout_temb is not None else temb
            with enable_lora((self.norm1.linear, self.norm1_context.linear), on=True, off=False):
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb_for_layout)

                norm_layout_hidden_states, layout_gate_msa, layout_shift_mlp, layout_scale_mlp, layout_gate_mlp = self.norm1_context(
                    layout_hidden_states, emb=temb_for_layout
                )
                
            bsz, max_objs = norm_layout_hidden_states.shape[:2]
            img_len = norm_hidden_states.shape[1]
            layout_attn_output = torch.zeros_like(layout_hidden_states)
            attn_output_add = torch.zeros_like(norm_hidden_states)
            img_add_cnt = torch.zeros((bsz, img_len)).to(dtype=attn_output_add.dtype, device=attn_output_add.device)
            img_weight_sum = torch.zeros((bsz, img_len, norm_hidden_states.shape[-1])).to(dtype=attn_output_add.dtype, device=attn_output_add.device)

            valid_mask = (layout_masks == 1)
            valid_indices = valid_mask.nonzero(as_tuple=False)

            if valid_indices.size(0) > 0:
                valid_bs = valid_indices[:, 0]
                valid_objs = valid_indices[:, 1]
                num_valid = valid_indices.size(0)

                layout_sigma_list = []
                layout_cosinesim_query_list = []
                
                for k in range(num_valid):
                    i = valid_bs[k].item()
                    j = valid_objs[k].item()
                    layout_sigma_item, layout_cosinesim_query_item = self.norm1_layout(
                        emb=layout_temb_list[i][0, j:j+1]
                    )
                    layout_sigma_list.append(layout_sigma_item)
                    layout_cosinesim_query_list.append(layout_cosinesim_query_item)
                
                layout_sigma = LAYOUT_SIGMA_SCALE * F.softplus(torch.cat(layout_sigma_list, dim=0)).to(layout_sigma_list[0].dtype)
                layout_cosinesim_query = torch.cat(layout_cosinesim_query_list, dim=0)

                all_scaled_outputs = []
                all_layout_outputs = []
                all_img_indices = []
                all_img_idxs_list = []
                ragged_img_hidden = []
                ragged_layout_hidden = []

                for k in range(num_valid):
                    i = valid_bs[k].item()
                    j = valid_objs[k].item()
                    img_idxs = img_idxs_list_list[i][j].to(norm_hidden_states.device).to(torch.int64)
                    all_img_idxs_list.append(img_idxs)
                    ragged_img_hidden.append(norm_hidden_states[i, img_idxs])
                    ragged_layout_hidden.append(norm_layout_hidden_states[i, j].unsqueeze(0))

                with enable_lora((
                    self.attn.to_q,
                    self.attn.to_k,
                    self.attn.to_v,
                    self.attn.add_q_proj,
                    self.attn.add_k_proj,
                    self.attn.add_v_proj,
                    self.attn.to_out[0],
                    self.attn.to_add_out,
                ), on=True, off=False):
                    ragged_img_attn_out, ragged_layout_attn_out = self.layout_nested_attn_processor(
                        self.attn,
                        hidden_states=ragged_img_hidden,
                        encoder_hidden_states=ragged_layout_hidden,
                        image_rotary_emb_list=layout_rotary_emb,
                    )

                for k in range(num_valid):
                    img_attn_out = ragged_img_attn_out[k]
                    layout_attn_out = ragged_layout_attn_out[k]
                    scaled_output = self.layout_forward(img_attn_out)
                    all_scaled_outputs.append(scaled_output)
                    all_layout_outputs.append(layout_attn_out.squeeze(0))

                    i = valid_bs[k].item()
                    img_idxs = all_img_idxs_list[k]
                    batch_indices = torch.full_like(img_idxs, i)
                    all_img_indices.append(torch.stack([batch_indices, img_idxs], dim=1))

                attn_output_add_list = []
                dim = all_scaled_outputs[0].shape[-1]
                for k in range(num_valid):
                    full_attn_output = torch.zeros(img_len, dim, 
                                                   device=norm_hidden_states.device, 
                                                   dtype=norm_hidden_states.dtype)
                    img_idxs = all_img_idxs_list[k]
                    src_output = all_scaled_outputs[k].to(dtype=full_attn_output.dtype)
                    full_attn_output.scatter_add_(
                        dim=0,
                        index=img_idxs.unsqueeze(-1).expand(-1, dim),
                        src=src_output
                    )
                    attn_output_add_list.append(full_attn_output)
                
                z_buffer_masks_2ch = None
                
                if occlusion is not None and bbox_mask is not None and all_scaled_outputs:
                    attn_output_add_stacked = torch.stack(attn_output_add_list, dim=0)
                    
                    layout_sigma_value = layout_sigma
                    
                    alpha = 1 - torch.exp(-layout_sigma_value)
                    
                    bbox_mask_len = len(bbox_mask) if isinstance(bbox_mask, list) else bbox_mask.shape[0]
                    bbox_masks_list = []
                    for k in range(num_valid):
                        obj_idx = valid_objs[k].item()
                        if obj_idx < bbox_mask_len:
                            bbox_masks_list.append(bbox_mask[obj_idx])
                        else:
                            bbox_masks_list.append(torch.zeros_like(bbox_mask[0]))
                    bbox_masks_stacked = torch.stack(bbox_masks_list, dim=0)
                    
                    z_buffer_mask_list = []
                    img_h = img_height if img_height is not None else int(img_len ** 0.5)
                    img_w = img_width if img_width is not None else int(img_len ** 0.5)
                    
                    for layout_idx in range(num_valid):
                        img_attn = attn_output_add_stacked[layout_idx]
                        query = layout_cosinesim_query[layout_idx]
                        
                        img_attn_normalized = F.normalize(img_attn, p=2, dim=-1)
                        query_normalized = F.normalize(query, p=2, dim=-1)
                        cosine_sim = (img_attn_normalized * query_normalized).sum(dim=-1)
                        
                        bbox_mask_1d = bbox_masks_stacked[layout_idx].squeeze(-1)
                        cosine_sim = cosine_sim * bbox_mask_1d
                        
                        z_mask_2d = cosine_sim.reshape(1, 1, img_h, img_w)
                        z_mask_2d = z_mask_2d.to(hidden_states.dtype)
                        
                        z_mask_2ch = self.mask_predictor(z_mask_2d)
                        z_mask_2ch_flat = z_mask_2ch.reshape(1, 2, img_h * img_w).permute(0, 2, 1)
                        z_buffer_mask_list.append(z_mask_2ch_flat)
                    
                    z_buffer_masks_inner = torch.cat(z_buffer_mask_list, dim=0)
                    z_buffer_masks_2ch = z_buffer_masks_inner 
                    
                    
                    transmittance_list = []
                    for layout_idx in range(num_valid):
                        j = valid_objs[layout_idx].item()
                        occluder_indices = occlusion[j] if j < len(occlusion) else []
                        
                        if len(occluder_indices) > 0:
                            occluder_valid_indices = []
                            for occ_idx in occluder_indices:
                                mask = (valid_objs == occ_idx)
                                if mask.any():
                                    occluder_valid_indices.append(mask.nonzero(as_tuple=True)[0].item())

                            if len(occluder_valid_indices) > 0:
                                occluder_bbox_masks = bbox_masks_stacked[occluder_valid_indices]
                                occluder_sigmas = layout_sigma_value[occluder_valid_indices]
                                mask_foreground = occluder_bbox_masks
                                accumulated_opacity = (mask_foreground * occluder_sigmas.unsqueeze(1)).sum(dim=0, keepdim=True)
                            else:
                                accumulated_opacity = torch.zeros(1, img_len, dim, device=bbox_masks_stacked.device, dtype=bbox_masks_stacked.dtype)
                        else:
                            accumulated_opacity = torch.zeros(1, img_len, dim, device=bbox_masks_stacked.device, dtype=bbox_masks_stacked.dtype)
                        
                        transmittance = torch.exp(-accumulated_opacity)
                        transmittance_list.append(transmittance.squeeze(0))
                    
                    transmittances = torch.stack(transmittance_list, dim=0)
                    

                    mask_final = bbox_masks_stacked
                    
                    weights = transmittances * alpha.unsqueeze(1) * mask_final
                    weighted_attn_outputs = weights * attn_output_add_stacked
                    
                    for k in range(num_valid):
                        i = valid_bs[k].item()
                        attn_output_add[i] = attn_output_add[i] + weighted_attn_outputs[k]
                        img_weight_sum[i] = img_weight_sum[i] + weights[k]
                        img_idxs = all_img_idxs_list[k]
                        img_add_cnt[i, img_idxs] = img_add_cnt[i, img_idxs] + 1
                    
                    layout_outputs = torch.stack(all_layout_outputs, dim=0)
                    layout_attn_output[valid_indices[:, 0], valid_indices[:, 1]] = layout_outputs
                    
                elif all_scaled_outputs:
                    z_buffer_masks_2ch = None
                    scaled_outputs = torch.cat(all_scaled_outputs, dim=0)
                    img_indices = torch.cat(all_img_indices, dim=0)
                    layout_outputs = torch.stack(all_layout_outputs, dim=0)

                    scaled_outputs = scaled_outputs.to(attn_output_add.dtype)

                    attn_output_add.view(-1, attn_output_add.size(-1)).scatter_add_(
                        dim=0,
                        index=(img_indices[:, 0] * img_len + img_indices[:, 1]).unsqueeze(-1).expand(-1, scaled_outputs.size(-1)),
                        src=scaled_outputs
                    )

                    img_add_cnt.view(-1).scatter_add_(
                        dim=0,
                        index=img_indices[:, 0] * img_len + img_indices[:, 1],
                        src=torch.ones(img_indices.size(0)).to(attn_output_add)
                    )

                    layout_attn_output[valid_indices[:, 0], valid_indices[:, 1]] = layout_outputs

            weight_sum_nonzero = (img_weight_sum.sum(dim=-1) > 0).float().unsqueeze(-1).to(attn_output_add.dtype)
            
            img_weight_sum_safe = (img_weight_sum + (img_weight_sum == 0).float() * 1e-8).to(attn_output_add.dtype)
            attn_output_weighted = attn_output_add / img_weight_sum_safe
            
            img_add_cnt_safe = (img_add_cnt.unsqueeze(-1) + (img_add_cnt.unsqueeze(-1) == 0).float()).to(attn_output_add.dtype)
            attn_output_simple = attn_output_add / img_add_cnt_safe
            
            attn_output_combined = weight_sum_nonzero * attn_output_weighted + (1 - weight_sum_nonzero) * attn_output_simple
            attn_output = gate_msa.unsqueeze(1) * attn_output_combined

            hidden_states = hidden_states + attn_output

            with enable_lora((self.ff.net[2],), on=True, off=False):
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                ff_output = self.ff(norm_hidden_states)
                ff_output = gate_mlp.unsqueeze(1) * ff_output

                hidden_states = hidden_states + ff_output

            layout_attn_output = layout_gate_msa.unsqueeze(1) * layout_attn_output
            layout_hidden_states = layout_hidden_states + layout_attn_output

            with enable_lora((self.ff_context.net[2],), on=True, off=False):
                norm_layout_hidden_states = self.norm2_context(layout_hidden_states)
                norm_layout_hidden_states = norm_layout_hidden_states * (1 + layout_scale_mlp[:, None]) + layout_shift_mlp[:, None]
                layout_ff_output = self.ff_context(norm_layout_hidden_states)
                layout_hidden_states = layout_hidden_states + layout_gate_mlp.unsqueeze(1) * layout_ff_output
                if layout_hidden_states.dtype == torch.float16:
                    layout_hidden_states = layout_hidden_states.clip(-65504, 65504)


        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states, layout_hidden_states, z_buffer_masks_2ch

class OcclusionFormerFluxTransformer2DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, FluxTransformer2DLoadersMixin, CacheMixin
):

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        block_type = "occlusion",
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                OcclusionFormerFluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    block_type=block_type if (i in LAYOUT_DUAL_STREAM_BLOCK_INDICES) else "default",
                )
                for i in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                OcclusionFormerFluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    block_type=block_type if (i in LAYOUT_SINGLE_STREAM_BLOCK_INDICES) else "default",
                )
                for i in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.block_type = block_type
        if is_occlusion_mode(self.block_type):
            self.layout_net = TextBoundingboxProjection(
                pooled_projection_dim=pooled_projection_dim,positive_len=self.inner_dim, out_dim=self.inner_dim
            )

        self._txt_ids_warning_shown = False
        self._img_ids_warning_shown = False

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def fuse_qkv_projections(self):
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedFluxAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        layout_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        enable_layout: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        if guidance is not None:
            layout_guidance_tensor = torch.full_like(guidance, LAYOUT_GUIDANCE_SCALE * 1000)
            layout_temb_for_block = self.time_text_embed(timestep, layout_guidance_tensor, pooled_projections)
        else:
            layout_temb_for_block = temb
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            if not self._txt_ids_warning_shown:
                logger.warning(
                    "Passing `txt_ids` 3d torch.Tensor is deprecated."
                    "Please remove the batch dimension and pass it as a 2d torch Tensor"
                )
                self._txt_ids_warning_shown = True
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            if not self._img_ids_warning_shown:
                logger.warning(
                    "Passing `img_ids` 3d torch.Tensor is deprecated."
                    "Please remove the batch dimension and pass it as a 2d torch Tensor"
                )
                self._img_ids_warning_shown = True
            img_ids = img_ids[0]

        multimodal_token_ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(multimodal_token_ids)

        if not isinstance(layout_kwargs, list):
            layout_kwargs = [layout_kwargs]
        object_box_batches, object_text_embed_batches, object_text_token_id_batches, object_presence_mask_batches, object_image_token_index_groups = [], [], [], [], []
        
        occlusion_parent_graph = []
        object_bbox_foreground_masks = []
        canvas_heights = []
        canvas_widths = []
        object_condition_time_embeds = []
        
        for ocf_layout_item in layout_kwargs:
            ocf_layout_payload = ocf_layout_item["layout"]
            object_box_batches.append(ocf_layout_payload["boxes"])
            object_condition_embed = ocf_layout_payload["text_embeddings"].to(dtype=hidden_states.dtype,device=hidden_states.device)
            object_text_embed_batches.append(object_condition_embed)
            object_text_token_id_batches.append(ocf_layout_payload["text_ids"])
            object_presence_mask_batches.append(ocf_layout_payload["masks"])
            object_image_token_index_groups.append(ocf_layout_payload["img_idxs_list"])

            object_condition_time_embeds.append(
                self.time_text_embed(timestep, object_condition_embed)
                if guidance is None
                else self.time_text_embed(timestep, layout_guidance_tensor, object_condition_embed)
            )
            
            if "occlusion" in ocf_layout_payload:
                occlusion_parent_graph.extend(ocf_layout_payload["occlusion"])
            if "bbox_mask" in ocf_layout_payload:
                object_bbox_foreground_masks.extend(ocf_layout_payload["bbox_mask"])
            if "img_height" in ocf_layout_payload:
                canvas_heights.append(ocf_layout_payload["img_height"])
            if "img_width" in ocf_layout_payload:
                canvas_widths.append(ocf_layout_payload["img_width"])

        object_boxes = torch.cat(object_box_batches, dim=0)
        object_condition_embeddings = torch.cat(object_text_embed_batches, dim=0)
        object_text_token_ids = torch.cat(object_text_token_id_batches, dim=0)
        object_presence_masks = torch.cat(object_presence_mask_batches, dim=0)
        layout_hidden_states = self.layout_net(
            boxes=object_boxes,
            masks=object_presence_masks,
            positive_embeddings=object_condition_embeddings,
        )

        object_rotary_emb_list = []
        active_object_mask = (object_presence_masks == 1)
        active_object_positions = active_object_mask.nonzero(as_tuple=False)
        if active_object_positions.size(0) > 0:
            active_object_batch_ids = active_object_positions[:, 0]
            active_object_ids = active_object_positions[:, 1]

            for k in range(active_object_positions.size(0)):
                sample_idx = active_object_batch_ids[k].item()
                object_idx = active_object_ids[k].item()
                object_img_token_indices = object_image_token_index_groups[sample_idx][object_idx]
                object_joint_token_ids = torch.cat(
                    (object_text_token_ids[sample_idx][object_idx].unsqueeze(0), img_ids[object_img_token_indices]),
                    dim=0,
                )
                object_rotary_emb_list.append(self.pos_embed(object_joint_token_ids))

        occlusion_zmask_outputs = []
        
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states, layout_hidden_states, z_buffer_masks_2ch = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                    layout_hidden_states,
                    object_presence_masks,
                    object_image_token_index_groups,
                    object_rotary_emb_list,
                    enable_layout,
                    layout_kwargs,
                    object_condition_time_embeds,
                    layout_temb_for_block,
                )

            else:
                encoder_hidden_states, hidden_states, layout_hidden_states, z_buffer_masks_2ch = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    layout_hidden_states=layout_hidden_states,
                    layout_masks=object_presence_masks,
                    img_idxs_list_list=object_image_token_index_groups,
                    layout_rotary_emb=object_rotary_emb_list,
                    enable_layout=enable_layout,
                    layout_kwargs=layout_kwargs,
                    layout_temb_list=object_condition_time_embeds,
                    layout_temb=layout_temb_for_block,
                )
            
            if z_buffer_masks_2ch is not None:
                occlusion_zmask_outputs.append(z_buffer_masks_2ch)
        

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states, layout_hidden_states, z_buffer_masks_2ch = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                    layout_hidden_states,
                    object_presence_masks,
                    object_image_token_index_groups,
                    object_rotary_emb_list,
                    enable_layout,
                    layout_kwargs,
                    object_condition_time_embeds,
                    layout_temb_for_block,
                )

            else:
                encoder_hidden_states, hidden_states, layout_hidden_states, z_buffer_masks_2ch = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    layout_hidden_states=layout_hidden_states,
                    layout_masks=object_presence_masks,
                    img_idxs_list_list=object_image_token_index_groups,
                    layout_rotary_emb=object_rotary_emb_list,
                    enable_layout=enable_layout,
                    layout_kwargs=layout_kwargs,
                    layout_temb_list=object_condition_time_embeds,
                    layout_temb=layout_temb_for_block,
                )
            
            if z_buffer_masks_2ch is not None:
                occlusion_zmask_outputs.append(z_buffer_masks_2ch)



        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output, occlusion_zmask_outputs if occlusion_zmask_outputs else None)

        return Transformer2DModelOutput(sample=output, z_buffer_masks=occlusion_zmask_outputs if occlusion_zmask_outputs else None)
