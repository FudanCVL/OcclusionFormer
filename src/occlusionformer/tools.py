from contextlib import contextmanager
from typing import Any, Iterable, List, Optional, Tuple
import types
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.tuners_utils import BaseTunerLayer
from PIL import Image

from diffusers.pipelines import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import logger
from diffusers.utils import logging
from diffusers.models.activations import FP32SiLU
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import CombinedTimestepLabelEmbeddings
from torch import Tensor
from ..utils import generate_distinct_palette


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.detach().zero_()
    return module


def is_occlusion_mode(block_type: str) -> bool:
    return block_type in {"occlusion", "layout"}


class AdaLayerNormZeroLayout(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, bias=True):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.query_liner = nn.Linear(embedding_dim, embedding_dim, bias=bias)

    def forward(
        self,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        query = self.query_liner(self.silu(emb))
        gate_sigma = self.linear(self.silu(emb))
        return gate_sigma, query


def _is_lora_linear(module: nn.Module) -> bool:
    return (
        isinstance(module, BaseTunerLayer)
        and hasattr(module, "base_layer")
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
    )


def _set_direct_lora_forward(module: nn.Module, scale: float) -> None:
    if not _is_lora_linear(module):
        return

    def _forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("adapter_names", None)
        forward_scale = kwargs.pop("scale", scale)
        if forward_scale is None:
            forward_scale = 1.0
        result = self.base_layer(x, *args, **kwargs)

        if forward_scale == 0.0:
            return result

        torch_result_dtype = result.dtype
        active_adapters = getattr(self, "active_adapters", [])
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]

        for adapter_name in active_adapters:
            if adapter_name not in self.lora_A or adapter_name not in self.lora_B:
                continue
            lora_A = self.lora_A[adapter_name]
            lora_B = self.lora_B[adapter_name]
            if hasattr(self, "_cast_input_dtype"):
                x_cast = self._cast_input_dtype(x, lora_A.weight.dtype)
            else:
                x_cast = x.to(lora_A.weight.dtype)
            dropout = self.lora_dropout[adapter_name]
            adapter_scaling = 1.0
            result = result + lora_B(lora_A(dropout(x_cast))) * adapter_scaling * forward_scale

        return result.to(torch_result_dtype)

    module.forward = types.MethodType(_forward, module)


@contextmanager
def enable_lora(modules: Iterable[nn.Module], on: bool = True, off: bool = False):
    mods = [m for m in modules if _is_lora_linear(m)]
    on_scale = 1.0 if on else 0.0
    off_scale = 1.0 if off else 0.0
    try:
        for m in mods:
            _set_direct_lora_forward(m, scale=on_scale)
        yield
    finally:
        for m in mods:
            _set_direct_lora_forward(m, scale=off_scale)


class FluxLayoutNestedAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxLayoutNestedAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    @staticmethod
    def _reshape_proj_to_heads(x: torch.Tensor, heads: int, head_dim: int) -> torch.Tensor:
        return x.reshape(-1, heads, head_dim).transpose(0, 1)

    @staticmethod
    def _flatten_heads_to_seq(x: torch.Tensor) -> torch.Tensor:
        return x.transpose(0, 1).reshape(x.shape[1], -1)

    @staticmethod
    def _apply_norm_per_item(norm_module, items: List[torch.Tensor]) -> List[torch.Tensor]:
        if norm_module is None:
            return items
        return [norm_module(item.unsqueeze(0)).squeeze(0) for item in items]

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        image_rotary_emb_list: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.FloatTensor:
        if isinstance(hidden_states, torch.Tensor):
            batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            if encoder_hidden_states is not None:
                encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                    batch_size, -1, attn.heads, head_dim
                ).transpose(1, 2)

                if attn.norm_added_q is not None:
                    encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
                if attn.norm_added_k is not None:
                    encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

                query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
                key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
                value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb

                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            if encoder_hidden_states is not None:
                encoder_hidden_states, hidden_states = (
                    hidden_states[:, : encoder_hidden_states.shape[1]],
                    hidden_states[:, encoder_hidden_states.shape[1] :],
                )

                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
                return hidden_states, encoder_hidden_states
            return hidden_states

        hidden_list = list(hidden_states)
        encoder_list = None if encoder_hidden_states is None else list(encoder_hidden_states)
        if encoder_list is not None and len(encoder_list) != len(hidden_list):
            raise ValueError("`hidden_states` and `encoder_hidden_states` should have the same batch/list length.")

        q_list, k_list, v_list = [], [], []
        context_lengths = []
        head_dim = attn.to_q.out_features // attn.heads

        for idx, hidden_item in enumerate(hidden_list):
            q = attn.to_q(hidden_item)
            k = attn.to_k(hidden_item)
            v = attn.to_v(hidden_item)

            q = self._reshape_proj_to_heads(q, attn.heads, head_dim)
            k = self._reshape_proj_to_heads(k, attn.heads, head_dim)
            v = self._reshape_proj_to_heads(v, attn.heads, head_dim)

            if encoder_list is not None:
                encoder_item = encoder_list[idx]
                eq = attn.add_q_proj(encoder_item)
                ek = attn.add_k_proj(encoder_item)
                ev = attn.add_v_proj(encoder_item)

                eq = self._reshape_proj_to_heads(eq, attn.heads, head_dim)
                ek = self._reshape_proj_to_heads(ek, attn.heads, head_dim)
                ev = self._reshape_proj_to_heads(ev, attn.heads, head_dim)

                context_lengths.append(eq.shape[1])
                q = torch.cat([eq, q], dim=1)
                k = torch.cat([ek, k], dim=1)
                v = torch.cat([ev, v], dim=1)
            else:
                context_lengths.append(0)

            q_list.append(q)
            k_list.append(k)
            v_list.append(v)

        q_list = self._apply_norm_per_item(attn.norm_q, q_list)
        k_list = self._apply_norm_per_item(attn.norm_k, k_list)
        if encoder_list is not None and attn.norm_added_q is not None:
            for idx, q in enumerate(q_list):
                context_len = context_lengths[idx]
                q_context = attn.norm_added_q(q[:, :context_len].unsqueeze(0)).squeeze(0)
                q_list[idx] = torch.cat([q_context, q[:, context_len:]], dim=1)
        if encoder_list is not None and attn.norm_added_k is not None:
            for idx, k in enumerate(k_list):
                context_len = context_lengths[idx]
                k_context = attn.norm_added_k(k[:, :context_len].unsqueeze(0)).squeeze(0)
                k_list[idx] = torch.cat([k_context, k[:, context_len:]], dim=1)

        if image_rotary_emb_list is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if len(image_rotary_emb_list) != len(q_list):
                raise ValueError("`image_rotary_emb_list` length should match ragged batch size.")
            for idx, rope in enumerate(image_rotary_emb_list):
                q_list[idx] = apply_rotary_emb(q_list[idx].unsqueeze(0), rope).squeeze(0)
                k_list[idx] = apply_rotary_emb(k_list[idx].unsqueeze(0), rope).squeeze(0)
        elif image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            for idx in range(len(q_list)):
                q_list[idx] = apply_rotary_emb(q_list[idx].unsqueeze(0), image_rotary_emb).squeeze(0)
                k_list[idx] = apply_rotary_emb(k_list[idx].unsqueeze(0), image_rotary_emb).squeeze(0)

        if len(q_list) == 1:
            out_item = F.scaled_dot_product_attention(
                q_list[0].unsqueeze(0),
                k_list[0].unsqueeze(0),
                v_list[0].unsqueeze(0),
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            ).squeeze(0)
            out_list_heads = [out_item]
        else:
            q_nested = torch.nested.as_nested_tensor([t.contiguous() for t in q_list])
            k_nested = torch.nested.as_nested_tensor([t.contiguous() for t in k_list])
            v_nested = torch.nested.as_nested_tensor([t.contiguous() for t in v_list])

            out_nested = F.scaled_dot_product_attention(
                q_nested, k_nested, v_nested, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            out_list_heads = list(out_nested.unbind())

        if encoder_list is not None:
            hidden_out, context_out = [], []
            for idx, out_item in enumerate(out_list_heads):
                out_item = self._flatten_heads_to_seq(out_item)
                context_len = context_lengths[idx]
                context_item = out_item[:context_len]
                hidden_item = out_item[context_len:]

                hidden_item = attn.to_out[0](hidden_item)
                hidden_item = attn.to_out[1](hidden_item)
                context_item = attn.to_add_out(context_item)

                hidden_out.append(hidden_item)
                context_out.append(context_item)
            return hidden_out, context_out

        return [self._flatten_heads_to_seq(out_item) for out_item in out_list_heads]


def encode_images(pipeline: FluxPipeline, images: Tensor):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids


def prepare_text_input(pipeline: FluxPipeline, prompts, max_sequence_length=512):
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids


class Layout:
    def __init__(self, conds, max_objs: int = 50, point_num: int = 36):
        self.hw = None
        self.max_objs = max_objs
        self.point_num = point_num

        self.cond_masks = torch.zeros(max_objs)
        self.categorys = [""] * max_objs
        self.texts = [""] * max_objs
        self.boxes = torch.zeros(max_objs, point_num * 2)

        self.set_conds(conds)

    def set_conds(self, annos):
        for i, anno in enumerate(annos):
            if i >= self.max_objs:
                break
            self.cond_masks[i] = 1
            self.categorys[i] = anno["category"]
            self.texts[i] = f"{anno['category']}"
            if anno.get("text"):
                self.texts[i] += f", {anno['text']}"

            bbox = anno["bbox"]
            hw = anno["hw"]
            self.boxes[i][0] = (bbox[0]) / hw[1]
            self.boxes[i][1] = (bbox[1]) / hw[0]
            self.boxes[i][2] = (bbox[0] + bbox[2]) / hw[1]
            self.boxes[i][3] = (bbox[1] + bbox[3]) / hw[0]

        self.dense_sample()

    def dense_sample(self):
        point_per_line = int(math.sqrt(self.point_num))
        for i in range(self.max_objs):
            if self.cond_masks[i] == 0:
                break
            x1, y1, x2, y2 = self.boxes[i][:4]
            step_x = (x2 - x1) / (point_per_line - 1)
            step_y = (y2 - y1) / (point_per_line - 1)
            for u in range(point_per_line):
                for v in range(point_per_line):
                    now_idx = (u * point_per_line + v) * 2
                    self.boxes[i][now_idx] = x1 + u * step_x
                    self.boxes[i][now_idx + 1] = y1 + v * step_y

    def show_layout_on_image(self, image: Image.Image) -> Image.Image:
        def draw(img_bgr: np.ndarray, boxes: torch.Tensor, labels: list[str]) -> np.ndarray:
            h, w, _ = img_bgr.shape
            _, bgr = generate_distinct_palette(self.max_objs)
            font = cv2.FONT_HERSHEY_SIMPLEX
            label_color = (255, 255, 255)

            for i in range(len(boxes)):
                color = bgr[i % len(bgr)] if bgr else (0, 255, 0)
                (x1, y1), (x2, y2) = boxes[i][:2], boxes[i][-2:]
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

                label = labels[i]
                sz, _ = cv2.getTextSize(label, font, 0.5, 1)
                cv2.rectangle(img_bgr, (x1, y1 - sz[1] - 10), (x1 + sz[0], y1), color, -1)
                cv2.putText(img_bgr, label, (x1, y1 - 7), font, 0.5, label_color, 1)

            return img_bgr

        if isinstance(image, Image.Image):
            arr = np.array(image)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            bgr = draw(bgr, self.boxes.clone(), self.categorys)
            return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        raise NotImplementedError("Only PIL.Image is supported here.")


def bbox_to_mask(bbox: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    mask = torch.zeros((latent_h, latent_w), device=bbox.device)
    (x1, y1), (x2, y2) = bbox[:2], bbox[-2:]
    if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
        return mask
    x1, y1, x2, y2 = int(x1 * latent_w), int(y1 * latent_h), int(x2 * latent_w), int(y2 * latent_h)
    if x1 == x2:
        x2 = x1 + 1
    if y1 == y2:
        y2 = y1 + 1
    mask[y1:y2, x1:x2] = 1
    return mask


def get_layout_idxslist(layout_boxes: torch.Tensor, latent_hw: tuple[int, int]):
    latent_h, latent_w = latent_hw
    img_idxs_list = []
    for obj_i in range(layout_boxes.shape[0]):
        mask_obj = bbox_to_mask(layout_boxes[obj_i], latent_h, latent_w)
        img_idxs = mask_obj.view(-1).nonzero().to(torch.int).view(-1)
        img_idxs_list.append(img_idxs)
    return img_idxs_list


def get_text_ids(layout_boxes: torch.Tensor, latent_h: int, latent_w: int) -> torch.Tensor:
    max_objs = layout_boxes.shape[0]
    text_ids = torch.zeros((max_objs, 3))
    for obj_i in range(max_objs):
        (x1, y1), (x2, y2) = layout_boxes[obj_i][:2], layout_boxes[obj_i][-2:]
        x1, y1, x2, y2 = int(x1 * latent_w), int(y1 * latent_h), int(x2 * latent_w), int(y2 * latent_h)
        text_ids[obj_i, 1] = (y1 + y2) / 2
        text_ids[obj_i, 2] = (x1 + x2) / 2
    return text_ids


def encode_layout_flux(layout: Layout, pipe: FluxPipeline, latent_hw: tuple[int, int]):
    prev_level = logger.level
    try:
        logger.setLevel(logging.ERROR)
        if layout.cond_masks.sum() == 0:
            return None

        device = pipe.device
        with torch.no_grad():
            text_embeddings = pipe._get_clip_prompt_embeds(
                prompt=layout.texts, device=device, num_images_per_prompt=1
            )

        text_ids = get_text_ids(layout.boxes, latent_hw[0], latent_hw[1])
        img_idxs_list = get_layout_idxslist(layout.boxes, latent_hw)

        boxes = layout.boxes.unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)
        text_embeddings = text_embeddings.unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)
        text_ids = text_ids.unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)
        masks = layout.cond_masks.unsqueeze(0).to(device=device, dtype=text_embeddings.dtype)

        layout_kwargs = {
            "layout": {
                "boxes": boxes,
                "text_embeddings": text_embeddings,
                "text_ids": text_ids,
                "masks": masks,
                "img_idxs_list": img_idxs_list,
            }
        }
        return layout_kwargs
    finally:
        logger.setLevel(prev_level)


def get_fourier_embeds_from_boundingbox(embed_dim, box, position_dim):
    batch_size, num_boxes = box.shape[:2]

    emb = 100 ** (torch.arange(embed_dim) / embed_dim)
    emb = emb[None, None, None].to(device=box.device, dtype=box.dtype)
    emb = emb * box.unsqueeze(-1)

    emb = torch.stack((emb.sin(), emb.cos()), dim=-1)
    emb = emb.permute(0, 1, 3, 4, 2).reshape(batch_size, num_boxes, position_dim)

    return emb


class PixArtAlphaTextProjection(nn.Module):
    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class TextBoundingboxProjection(nn.Module):
    def __init__(self, pooled_projection_dim, positive_len, out_dim, fourier_freqs=8):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim

        self.fourier_embedder_dim = fourier_freqs
        self.position_dim = fourier_freqs * 2 * 36 * 2

        if isinstance(out_dim, tuple):
            out_dim = out_dim[0]

        self.text_embedder = PixArtAlphaTextProjection(pooled_projection_dim, positive_len, act_fn="silu")
        self.linears = PixArtAlphaTextProjection(
            in_features=self.positive_len + self.position_dim,
            hidden_size=out_dim // 2,
            out_features=out_dim,
            act_fn="silu",
        )

    def forward(
        self,
        boxes,
        masks,
        positive_embeddings,
    ):
        masks = masks.unsqueeze(-1)
        xyxy_embedding = get_fourier_embeds_from_boundingbox(self.fourier_embedder_dim, boxes, self.position_dim)
        xyxy_embedding = xyxy_embedding * masks
        positive_embeddings = self.text_embedder(positive_embeddings)
        positive_embeddings = positive_embeddings * masks
        objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))

        return objs
