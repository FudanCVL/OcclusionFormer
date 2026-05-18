import json
import time
import io
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from PIL import Image, ImageDraw
import colorsys
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

import numpy as np

from src.occlusionformer.tools import Layout
from src.occlusionformer.transformer import OcclusionFormerFluxTransformer2DModel
from src.occlusionformer.inference import inference
from src.utils import generate_distinct_palette, xyxy2xywh
from diffusers.pipelines import FluxPipeline

# ---------------------- Config ----------------------
st.set_page_config(page_title="InstanceAssemble Demo (textual-only)", layout="wide")

MAX_OBJS = 50
default_seed = 42
default_height = 1024
default_width = 1024
PALETTE, bgr = generate_distinct_palette(MAX_OBJS)

# Keys for st.session_state
SS_BOXES = "boxes"                 
SS_ADDED_SET = "added_bbox_set"    
SS_CANVAS_KEY = "canvas_key"       
SS_PROMPT = "global_prompt"
SS_SEED = "seed"
SS_HEIGHT = "height"
SS_WIDTH = "width"
SS_PENDING_EXAMPLE = "pending_example"  
SS_PIPE = "loaded_pipe"
SS_LT = "loaded_layout_transformer"
SS_LOADED = "is_model_loaded"
SS_LOADED_CFG = "loaded_cfg_triple"
SS_CANVAS_PNG = "canvas_png_bytes"
SS_AUTO_SAVE_LAYOUT = "auto_save_layout_once"

# ---------------------- Preset Examples (for bottom Examples column) ----------------------
def load_presets_from_demo(demo_dir: str = "./examples"):
    presets = []
    demo_path = Path(demo_dir)

    for json_file in sorted(demo_path.glob("*.json"), key=lambda p: p.name):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        boxes = []
        for box in data.get("annos", []):
            boxes.append({
                "bbox": box["bbox"],
                "category": (str(box.get("category_name", "")) if box.get("category_name", "") is not None else "").strip(),
                "caption": box.get("caption", ""),
                "occludes": box.get("occludes", ""),
            })

        preset = {
            "name": json_file.stem,
            "height": data.get("height", default_height),
            "width": data.get("width", default_width),
            "prompt": data.get("prompt", data.get("caption", "")),
            "annos": boxes,
            "seed": data.get("seed", default_seed)
        }
        presets.append(preset)

    return presets

PRESET_CASES = load_presets_from_demo("./examples")


@dataclass
class GenSettings:
    model_type: str
    model_path: str
    ckpt_path: str
    gen_H: int
    gen_W: int
    steps: int
    guidance_scale: float
    enable_layout: bool
    grounding_ratio: float
    seed: int
    canvas_w: int
    stroke_width: int
    device: torch.device
    dtype: torch.dtype = torch.bfloat16

def _rgba(hex_rgb: str, alpha: float) -> str:
    if hex_rgb.startswith("#") and len(hex_rgb) == 7:
        return f"{hex_rgb}{int(alpha * 255):02x}"
    return hex_rgb

def _hex_to_rgba_str(hex_rgb: str, a: float = 0.12) -> str:
    """#RRGGBB -> 'rgba(r,g,b,a)'"""
    h = hex_rgb.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _compute_occlusion_layers(boxes: List[Dict]) -> List[int]:
    num_objs = len(boxes)
    if num_objs == 0:
        return []

    graph = [set() for _ in range(num_objs)]
    indegree = [0 for _ in range(num_objs)]

    for src_idx, box in enumerate(boxes):
        occludes_str = str(box.get("occludes", "")).strip()
        if not occludes_str:
            continue
        try:
            dst_ids = [int(x.strip()) for x in occludes_str.split(",") if x.strip()]
        except ValueError:
            continue
        for dst_id in dst_ids:
            dst_idx = dst_id - 1
            if dst_idx < 0 or dst_idx >= num_objs or dst_idx == src_idx:
                continue
            if dst_idx not in graph[src_idx]:
                graph[src_idx].add(dst_idx)
                indegree[dst_idx] += 1

    layers = [0 for _ in range(num_objs)]
    queue = [i for i in range(num_objs) if indegree[i] == 0]
    queue.sort()

    head = 0
    visited = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        visited += 1
        for nxt in sorted(graph[node]):
            layers[nxt] = max(layers[nxt], layers[node] + 1)
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if visited < num_objs:
        for idx in range(num_objs):
            direct_occluders = 0
            for src in range(num_objs):
                if idx in graph[src]:
                    direct_occluders += 1
            layers[idx] = max(layers[idx], direct_occluders)

    return layers


def _parse_occlusion_edges(boxes: List[Dict]) -> List[Tuple[int, int]]:
    edges = set()
    num_objs = len(boxes)
    for src_idx, box in enumerate(boxes):
        occludes_str = str(box.get("occludes", "")).strip()
        if not occludes_str:
            continue
        try:
            dst_ids = [int(x.strip()) for x in occludes_str.split(",") if x.strip()]
        except ValueError:
            continue
        for dst_id in dst_ids:
            dst_idx = dst_id - 1
            if 0 <= dst_idx < num_objs and dst_idx != src_idx:
                edges.add((src_idx, dst_idx))
    return sorted(list(edges))


def _bbox_intersection_xyxy(
    bbox_a: List[int], bbox_b: List[int]
) -> Optional[Tuple[int, int, int, int]]:
    ax1, ay1, ax2, ay2 = [int(v) for v in bbox_a]
    bx1, by1, bx2, by2 = [int(v) for v in bbox_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def _subtract_one_rect(
    base: Tuple[int, int, int, int], cut: Tuple[int, int, int, int]
) -> List[Tuple[int, int, int, int]]:
    inter = _bbox_intersection_xyxy(list(base), list(cut))
    if inter is None:
        return [base]

    x1, y1, x2, y2 = base
    ix1, iy1, ix2, iy2 = inter
    parts = []

    if iy1 > y1:
        parts.append((x1, y1, x2, iy1))
    if iy2 < y2:
        parts.append((x1, iy2, x2, y2))
    if ix1 > x1:
        parts.append((x1, iy1, ix1, iy2))
    if ix2 < x2:
        parts.append((ix2, iy1, x2, iy2))

    return [r for r in parts if r[2] > r[0] and r[3] > r[1]]


def _subtract_rectangles(
    base: Tuple[int, int, int, int], cuts: List[Tuple[int, int, int, int]]
) -> List[Tuple[int, int, int, int]]:
    remain = [base]
    for cut in cuts:
        next_remain = []
        for r in remain:
            next_remain.extend(_subtract_one_rect(r, cut))
        remain = next_remain
        if not remain:
            break
    return remain


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def _subtract_intervals(
    base: Tuple[int, int], cuts: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    x1, x2 = base
    if x2 <= x1:
        return []
    if not cuts:
        return [(x1, x2)]

    merged = _merge_intervals(
        [(max(x1, s), min(x2, e)) for s, e in cuts if min(x2, e) > max(x1, s)]
    )
    if not merged:
        return [(x1, x2)]

    out = []
    cur = x1
    for s, e in merged:
        if s > cur:
            out.append((cur, s))
        cur = max(cur, e)
    if cur < x2:
        out.append((cur, x2))
    return out


def _pick_label_text_color(hex_rgb: str) -> str:
    h = hex_rgb.lstrip("#")
    if len(h) != 6:
        return "#111111"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    yiq = (r * 299 + g * 587 + b * 114) / 1000
    return "#111111" if yiq >= 140 else "#f5f5f5"


def _estimate_text_width(text: str, font_size: int = 12) -> float:
    return max(12.0, len(text) * font_size * 0.62)


def _render_occlusionformer_header() -> None:
    logo_path = Path("./occlusionformer.png")
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
        st.markdown(
            f"""
<div style="display:flex;align-items:center;gap:14px;margin:0 0 6px 0;">
  <img src="data:image/png;base64,{logo_b64}"
       style="width:56px;height:56px;object-fit:contain;image-rendering:crisp-edges;"/>
  <div style="font-size:3rem;font-weight:700;line-height:1;color:inherit;">OcclusionFormer</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.title("OcclusionFormer")


def _canvas_image_to_png_bytes(canvas_image_data: np.ndarray) -> bytes:
    if canvas_image_data.dtype != np.uint8:
        arr = np.clip(canvas_image_data, 0, 255).astype(np.uint8)
    else:
        arr = canvas_image_data
    image = Image.fromarray(arr, mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_initial_fabric_from_boxes(
    boxes: List[Dict], view_W: int, view_H: int, gen_W: int, gen_H: int, stroke_w: int
) -> Dict:
    if gen_W <= 0 or gen_H <= 0 or view_W <= 0 or view_H <= 0:
        return {"objects": []}

    sx = view_W / float(gen_W)
    sy = view_H / float(gen_H)

    layers = _compute_occlusion_layers(boxes)
    max_layer = max(layers) if layers else 0

    visible_fill_alpha = 0.24
    occluded_fill_alpha = 0.09
    visible_stroke_alpha = 1.0
    occluded_stroke_alpha = 0.15

    occluder_map: Dict[int, List[int]] = {i: [] for i in range(len(boxes))}
    occludes_map: Dict[int, List[int]] = {i: [] for i in range(len(boxes))}
    explicit_edges = _parse_occlusion_edges(boxes)
    for occluder_idx, occluded_idx in explicit_edges:
        if 0 <= occluder_idx < len(boxes) and 0 <= occluded_idx < len(boxes):
            occluder_map[occluded_idx].append(occluder_idx)
            occludes_map[occluder_idx].append(occluded_idx)

    objects = []
    for box_idx, b in enumerate(boxes):
        x1, y1, x2, y2 = b["bbox"]
        base_rect = (int(x1), int(y1), int(x2), int(y2))
        color = b.get("color", "#000000")
        has_explicit_relation = bool(occluder_map.get(box_idx) or occludes_map.get(box_idx))

        if not has_explicit_relation:
            left = float(base_rect[0]) * sx
            top = float(base_rect[1]) * sy
            width = float(base_rect[2] - base_rect[0]) * sx
            height = float(base_rect[3] - base_rect[1]) * sy
            objects.append({
                "type": "rect",
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "fill": _hex_to_rgba_str(color, 0.12),
                "stroke": color,
                "strokeWidth": float(stroke_w),
                "selectable": False,
                "evented": False,
            })

            category = str(b.get("category", "")).strip()
            if category:
                label_padding = 6
                label_left = float(base_rect[0] + label_padding) * sx
                label_top = float(base_rect[1] + label_padding) * sy
                label_width = max(16.0, _estimate_text_width(category, font_size=12) + 8.0)
                max_right = float(view_W) - 2.0
                if label_left + label_width > max_right:
                    label_left = max(2.0, max_right - label_width)
                label_height = 16.0
                label_fill = color
                label_text_color = _pick_label_text_color(color)
                objects.append({
                    "type": "rect",
                    "left": label_left,
                    "top": label_top,
                    "width": label_width,
                    "height": label_height,
                    "fill": label_fill,
                    "stroke": "",
                    "strokeWidth": 0.0,
                    "selectable": False,
                    "evented": False,
                })
                objects.append({
                    "type": "text",
                    "text": category,
                    "left": label_left + 4.0,
                    "top": label_top + 1.0,
                    "fontSize": 12,
                    "fontWeight": "600",
                    "fill": label_text_color,
                    "selectable": False,
                    "evented": False,
                })
            continue

        layer = layers[box_idx] if box_idx < len(layers) else 0
        layer_norm = float(layer) / float(max_layer) if max_layer > 0 else 0.0

        cur_occluded_fill_alpha = min(0.22, occluded_fill_alpha + 0.04 * layer_norm)
        cur_occluded_stroke_alpha = min(0.60, occluded_stroke_alpha + 0.10 * layer_norm)

        # 1) Base rectangle uses lower alpha: this is the occluded appearance.
        left  = float(base_rect[0]) * sx
        top   = float(base_rect[1]) * sy
        width = float(base_rect[2] - base_rect[0]) * sx
        height= float(base_rect[3] - base_rect[1]) * sy
        objects.append({
            "type": "rect",
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "fill": _hex_to_rgba_str(color, cur_occluded_fill_alpha),
            "stroke": _hex_to_rgba_str(color, cur_occluded_stroke_alpha),
            "strokeWidth": float(stroke_w),
            "selectable": False,
            "evented": False,
        })

        # 2) Compute overlap cuts, then redraw only non-occluded regions with higher alpha.
        cuts: List[Tuple[int, int, int, int]] = []
        for occ_idx in occluder_map.get(box_idx, []):
            inter = _bbox_intersection_xyxy(boxes[occ_idx]["bbox"], b["bbox"])
            if inter is not None:
                cuts.append(inter)

        visible_rects = _subtract_rectangles(base_rect, cuts) if cuts else [base_rect]
        for vx1, vy1, vx2, vy2 in visible_rects:
            objects.append({
                "type": "rect",
                "left": float(vx1) * sx,
                "top": float(vy1) * sy,
                "width": float(vx2 - vx1) * sx,
                "height": float(vy2 - vy1) * sy,
                "fill": _hex_to_rgba_str(color, visible_fill_alpha),
                "stroke": "",
                "strokeWidth": 0.0,
                "selectable": False,
                "evented": False,
            })

        # 3) Border handling: redraw only visible border segments with higher alpha.
        top_cuts = []
        bot_cuts = []
        left_cuts = []
        right_cuts = []
        for ix1, iy1, ix2, iy2 in cuts:
            if iy1 <= base_rect[1] <= iy2:
                top_cuts.append((ix1, ix2))
            if iy1 <= base_rect[3] <= iy2:
                bot_cuts.append((ix1, ix2))
            if ix1 <= base_rect[0] <= ix2:
                left_cuts.append((iy1, iy2))
            if ix1 <= base_rect[2] <= ix2:
                right_cuts.append((iy1, iy2))

        top_vis = _subtract_intervals((base_rect[0], base_rect[2]), top_cuts)
        bot_vis = _subtract_intervals((base_rect[0], base_rect[2]), bot_cuts)
        left_vis = _subtract_intervals((base_rect[1], base_rect[3]), left_cuts)
        right_vis = _subtract_intervals((base_rect[1], base_rect[3]), right_cuts)

        stroke_color = _hex_to_rgba_str(color, visible_stroke_alpha)
        for x_start, x_end in top_vis:
            if x_end > x_start:
                objects.append({
                    "type": "line",
                    "x1": float(x_start) * sx,
                    "y1": float(base_rect[1]) * sy,
                    "x2": float(x_end) * sx,
                    "y2": float(base_rect[1]) * sy,
                    "stroke": stroke_color,
                    "strokeWidth": float(stroke_w),
                    "selectable": False,
                    "evented": False,
                })
        for x_start, x_end in bot_vis:
            if x_end > x_start:
                objects.append({
                    "type": "line",
                    "x1": float(x_start) * sx,
                    "y1": float(base_rect[3]) * sy,
                    "x2": float(x_end) * sx,
                    "y2": float(base_rect[3]) * sy,
                    "stroke": stroke_color,
                    "strokeWidth": float(stroke_w),
                    "selectable": False,
                    "evented": False,
                })
        for y_start, y_end in left_vis:
            if y_end > y_start:
                objects.append({
                    "type": "line",
                    "x1": float(base_rect[0]) * sx,
                    "y1": float(y_start) * sy,
                    "x2": float(base_rect[0]) * sx,
                    "y2": float(y_end) * sy,
                    "stroke": stroke_color,
                    "strokeWidth": float(stroke_w),
                    "selectable": False,
                    "evented": False,
                })
        for y_start, y_end in right_vis:
            if y_end > y_start:
                objects.append({
                    "type": "line",
                    "x1": float(base_rect[2]) * sx,
                    "y1": float(y_start) * sy,
                    "x2": float(base_rect[2]) * sx,
                    "y2": float(y_end) * sy,
                    "stroke": stroke_color,
                    "strokeWidth": float(stroke_w),
                    "selectable": False,
                    "evented": False,
                })

        category = str(b.get("category", "")).strip()
        if category:
            label_padding = 6
            label_left = float(base_rect[0] + label_padding) * sx
            label_top = float(base_rect[1] + label_padding) * sy
            label_width = max(16.0, _estimate_text_width(category, font_size=12) + 8.0)
            max_right = float(view_W) - 2.0
            if label_left + label_width > max_right:
                label_left = max(2.0, max_right - label_width)
            label_height = 16.0
            label_x1 = int(round(label_left / max(sx, 1e-6)))
            label_y1 = int(round(label_top / max(sy, 1e-6)))
            label_w_in_layout = int(round(label_width / max(sx, 1e-6)))
            label_h_in_layout = int(round(label_height / max(sy, 1e-6)))
            label_rect = (
                label_x1,
                label_y1,
                label_x1 + max(1, label_w_in_layout),
                label_y1 + max(1, label_h_in_layout),
            )
            is_label_in_occluded_region = False
            for cx1, cy1, cx2, cy2 in cuts:
                if _bbox_intersection_xyxy(list(label_rect), [cx1, cy1, cx2, cy2]) is not None:
                    is_label_in_occluded_region = True
                    break
            label_alpha = cur_occluded_stroke_alpha if is_label_in_occluded_region else visible_stroke_alpha
            label_fill = _hex_to_rgba_str(color, label_alpha)
            label_text_color = _pick_label_text_color(color)
            objects.append({
                "type": "rect",
                "left": label_left,
                "top": label_top,
                "width": label_width,
                "height": label_height,
                "fill": label_fill,
                "stroke": "",
                "strokeWidth": 0.0,
                "selectable": False,
                "evented": False,
            })
            objects.append({
                "type": "text",
                "text": category,
                "left": label_left + 4.0,
                "top": label_top + 1.0,
                "fontSize": 12,
                "fontWeight": "600",
                "fill": label_text_color,
                "selectable": False,
                "evented": False,
            })

    return {"objects": objects}

def scale_rect_to_gen_space(o: dict, sx: float, sy: float) -> Tuple[int, int, int, int]:
    """Scale a rect (fabric.js object) from canvas coords into generation-space coords."""
    left = float(o.get("left", 0))
    top = float(o.get("top", 0))
    w = float(o.get("width", 0)) * float(o.get("scaleX", 1))
    h = float(o.get("height", 0)) * float(o.get("scaleY", 1))
    x1, y1 = int(round(left * sx)), int(round(top * sy))
    x2, y2 = int(round((left + w) * sx)), int(round((top + h) * sy))
    return x1, y1, x2, y2

def ensure_session_state():
    if SS_BOXES not in st.session_state:
        st.session_state[SS_BOXES] = []
    if SS_ADDED_SET not in st.session_state:
        st.session_state[SS_ADDED_SET] = set()
    if SS_CANVAS_KEY not in st.session_state:
        st.session_state[SS_CANVAS_KEY] = 0
    if SS_PROMPT not in st.session_state:
        st.session_state[SS_PROMPT] = ""
    if SS_SEED not in st.session_state:
        st.session_state[SS_SEED] = default_seed
    if SS_HEIGHT not in st.session_state:
        st.session_state[SS_HEIGHT] = default_height
    if SS_WIDTH not in st.session_state:
        st.session_state[SS_WIDTH] = default_width
    if SS_PENDING_EXAMPLE not in st.session_state:
        st.session_state[SS_PENDING_EXAMPLE] = None
    if SS_PIPE not in st.session_state:
        st.session_state[SS_PIPE] = None
    if SS_LT not in st.session_state:
        st.session_state[SS_LT] = None
    if SS_LOADED not in st.session_state:
        st.session_state[SS_LOADED] = False
    if SS_LOADED_CFG not in st.session_state:
        st.session_state[SS_LOADED_CFG] = None
    if SS_CANVAS_PNG not in st.session_state:
        st.session_state[SS_CANVAS_PNG] = None
    if SS_AUTO_SAVE_LAYOUT not in st.session_state:
        st.session_state[SS_AUTO_SAVE_LAYOUT] = False


def next_color_for(index: int) -> str:
    return PALETTE[index % len(PALETTE)]

def _reset_on_model_type_change():
    mt = st.session_state["model_type"]
    if mt == "Flux.1-dev":
        st.session_state["steps"] = 28
        st.session_state["ckpt_path"] = "./ckpt/occlusionformer"


def _build_flux_layout_transformer_from_ckpt(model_path: str, ckpt_dir: Path, dtype: torch.dtype, device: torch.device):
    # Load directly with from_pretrained to avoid an extra copy.
    layout_transformer = OcclusionFormerFluxTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
        block_type="occlusion",
    )

    lora_file = ckpt_dir / "lora.safetensors"
    if not lora_file.exists():
        raise FileNotFoundError(f"Missing LoRA: {lora_file}")
    lora_sd, alphas = FluxPipeline.lora_state_dict(
        pretrained_model_name_or_path_or_dict=str(ckpt_dir),
        weight_name=lora_file.name,
        return_alphas=True,
    )
    FluxPipeline.load_lora_into_transformer(state_dict=lora_sd, network_alphas=alphas, transformer=layout_transformer)

    layout_file = ckpt_dir / "occ.pth"
    if not layout_file.exists():
        raise FileNotFoundError(f"Missing layout state dict: {layout_file}")
    
    # Load layout weights and keep dtype consistent.
    layout_state_dict = torch.load(layout_file, map_location="cpu", weights_only=False)
    # Cast layout weights to target dtype.
    layout_state_dict = {k: v.to(dtype) if v.dtype.is_floating_point else v 
                         for k, v in layout_state_dict.items()}
    missing, unexpected = layout_transformer.load_state_dict(layout_state_dict, strict=False)

    layout_missing = [
        k for k in missing
        if "layout" in k or "mask_predictor" in k or "norm1_layout" in k
    ]
    layout_unexpected = [
        k for k in unexpected
        if "layout" in k or "mask_predictor" in k or "norm1_layout" in k
    ]

    print("layout missing keys:", layout_missing[:100])
    print("layout unexpected keys:", layout_unexpected[:100])
    print("num layout missing:", len(layout_missing))
    print("num layout unexpected:", len(layout_unexpected))

    # _zero_out_lora_scales(layout_transformer)
    # Ensure the whole model uses a consistent dtype.
    return layout_transformer.to(dtype=dtype, device=device)


@st.cache_resource(show_spinner=False)
def load_flux_bundle(
    model_path: str,
    ckpt_path: str,
    dtype: torch.dtype,
    device: torch.device,
):
    """Load Flux backbone and (if ckpt_path valid) layout transformer with weights."""
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    pipe.transformer.to('cpu')
    ckpt_dir = Path(ckpt_path) if ckpt_path else None
    if ckpt_dir and ckpt_dir.exists():
        layout_transformer = _build_flux_layout_transformer_from_ckpt(model_path, ckpt_dir, dtype, device)
    return pipe, layout_transformer


def sidebar_controls() -> GenSettings:
    with st.sidebar:
        st.subheader("Model Setting")
        model_type = st.selectbox("Backbone Model", ["Flux.1-dev", "Flux.1-schnell"], index=0, key="model_type", on_change=_reset_on_model_type_change)
        _reset_on_model_type_change()
        model_path = st.text_input("model_id/model_path（Backbone）", value="/home/volume_shared/share_datasets/data_nvme_3/pretrained_weights/FLUX.1-dev/")
        ckpt_path = st.text_input("ckpt_path", value=st.session_state["ckpt_path"], help="including pytorch_lora_weights.safetensors and layout.pth")
        
        cfg_triple = (
            model_type,
            model_path.strip(),
            ckpt_path.strip(),
        )

        if st.session_state[SS_LOADED] and st.session_state[SS_LOADED_CFG] != cfg_triple:
            st.info("model settings changed: please click Load model below to reload.")

        if st.button("📦 Load model", use_container_width=True):
            try:
                load_flux_bundle.clear()
                if st.session_state[SS_PIPE] is not None:
                    st.session_state[SS_PIPE].to('cpu')
                    del st.session_state[SS_PIPE]
                if st.session_state[SS_LT] is not None:
                    st.session_state[SS_LT].to('cpu')
                    del st.session_state[SS_LT]
                torch.cuda.empty_cache()
                
                st.session_state[SS_PIPE] = None
                st.session_state[SS_LT] = None
                st.session_state[SS_LOADED] = False

                _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                _dtype = torch.bfloat16

                pipe, layout_transformer = load_flux_bundle(
                    model_path.strip(),
                    ckpt_path.strip(),
                    _dtype,
                    _device,
                )

                st.session_state[SS_PIPE] = pipe
                st.session_state[SS_LT] = layout_transformer
                st.session_state[SS_LOADED_CFG] = cfg_triple
                st.session_state[SS_LOADED] = True
                st.success("Model loaded.")
            except Exception as e:
                st.error(f"Model loading failed.")

        st.subheader("Inference Setting")
        col_hw1, col_hw2 = st.columns(2)
        with col_hw1:
            gen_H = st.number_input("Generate Height", min_value=256, max_value=3096, value=st.session_state.get(SS_HEIGHT, default_height), step=64, on_change=st.rerun)
        with col_hw2:
            gen_W = st.number_input("Generate Width", min_value=256, max_value=3096, value=st.session_state.get(SS_WIDTH, default_width), step=64, on_change=st.rerun)

        steps = st.slider("num_inference_steps", 5, 100, st.session_state["steps"], 1)
        guidance_scale = st.slider("guidance_scale", 0.0, 25.0, 3.5, 0.1, help="Guidance scale after grounding steps")
        enable_layout = st.checkbox("enable_layout", value=True)
        grounding_ratio = st.slider("grounding_ratio", 0.0, 1.0, 0.3, 0.05, help="Apply layout control during the first X%% of the steps")
        seed = st.number_input("seed", value=st.session_state[SS_SEED], step=1)

        st.markdown("---")
        st.subheader("Canvas Setting")
        canvas_w = st.slider(
            "Canvas display width (px)",
            min_value=120, max_value=1000, value=400, step=10,
            help="Only affects the visualization size; the coordinates will be proportionally mapped to the generation resolution.",
            on_change=st.rerun
        )
        stroke_width = st.slider("Stroke width", 1, 6, 2, 1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        st.caption(f"Device: **{device}**；dtype: bfloat16")

    return GenSettings(
        model_type=model_type,
        model_path=model_path,
        ckpt_path=ckpt_path,
        gen_H=int(gen_H),
        gen_W=int(gen_W),
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        enable_layout=bool(enable_layout),
        grounding_ratio=float(grounding_ratio),
        seed=int(seed),
        canvas_w=int(canvas_w),
        stroke_width=int(stroke_width),
        device=device,
    )


def panel_canvas_and_table(settings: GenSettings) -> str:
    """Left column: canvas + Add Instance + editable table."""
    ensure_session_state()

    _render_occlusionformer_header()
    st.markdown(
        "<p style='font-size:16px; color:gray;'>Draw a box first ➜ click Add Instance ➜ enter the instance category and caption</p>",
        unsafe_allow_html=True
    )

    st.text_area(
        "Global Prompt",
        key=SS_PROMPT,
        height=88,
    )
    prompt = st.session_state[SS_PROMPT]

    # Visual canvas size (purely UI), then map to gen size.
    view_W = int(settings.canvas_w)
    _ratio = (view_W / settings.gen_W) if int(settings.gen_W) > 0 else 1.0
    view_H = int(round(settings.gen_H * _ratio))

    next_color = next_color_for(len(st.session_state[SS_BOXES]))
    rendered_bbox_png = st.session_state.get(SS_CANVAS_PNG, None)


    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("🧹 Clear canvas", type="secondary", width="stretch"):
            st.session_state[SS_CANVAS_KEY] += 1
            st.session_state[SS_BOXES].clear()
            st.session_state[SS_ADDED_SET].clear()
            st.session_state[SS_CANVAS_PNG] = None
            st.success("Cleared")
            st.rerun()
    with action_col2:
        st.download_button(
            "⬇️ Download Rendered BBox",
            data=rendered_bbox_png if rendered_bbox_png is not None else b"",
            file_name="rendered_bbox.png",
            mime="image/png",
            width="stretch",
            disabled=rendered_bbox_png is None,
        )

    initial_fabric = _build_initial_fabric_from_boxes(
        st.session_state[SS_BOXES],
        view_W=view_W,
        view_H=view_H,
        gen_W=settings.gen_W,
        gen_H=settings.gen_H,
        stroke_w=settings.stroke_width,
    )

    canvas = st_canvas(
        fill_color=_rgba(next_color, 0.12),
        stroke_color=next_color,
        background_color="#f7f7f9",
        update_streamlit=True,
        height=int(view_H),
        width=int(view_W),
        drawing_mode="rect",
        stroke_width=settings.stroke_width,
        key=f"canvas_{st.session_state[SS_CANVAS_KEY]}",
        initial_drawing=initial_fabric,
        display_toolbar=False,
    )

    if canvas is not None and canvas.image_data is not None:
        rendered_bbox_png = _canvas_image_to_png_bytes(canvas.image_data)
        st.session_state[SS_CANVAS_PNG] = rendered_bbox_png

    # extract last rect
    raw_last_rect = None
    if canvas and canvas.json_data:
        rect_objs = [
            o for o in canvas.json_data.get("objects", [])
                if o.get("type") == "rect" and o.get("selectable", True)  # Keep only selectable rects (newly drawn by user).
        ]
        if rect_objs:
            raw_last_rect = rect_objs[-1]

    sx, sy = settings.gen_W / float(view_W), settings.gen_H / float(view_H)

    if st.button("➕ Add Instance", type="primary", width='stretch'):
        if raw_last_rect is None:
            st.warning("Please draw a rectangle on the canvas first, then click Add Instance.")
        else:
            x1, y1, x2, y2 = scale_rect_to_gen_space(raw_last_rect, sx, sy)
            bbox_t = (x1, y1, x2, y2)
            if bbox_t in st.session_state[SS_ADDED_SET]:
                st.info("This rectangle has already been added (same coordinates), no need to add again.")
            else:
                color = next_color_for(len(st.session_state[SS_BOXES]))
                st.session_state[SS_BOXES].append({
                    "bbox": [x1, y1, x2, y2],
                    "category": "",
                    "caption": "",
                    "color": color,
                    "occludes": "",  # IDs of objects this instance occludes, comma-separated.
                })
                st.session_state[SS_ADDED_SET].add(bbox_t)
                st.success(f"Added an instance: {bbox_t}")
                st.rerun()

    if len(st.session_state[SS_BOXES]) > 0:
        st.subheader("✏️ Update Instance BBox")
        edit_cols = st.columns([1, 3, 1])
        with edit_cols[0]:
            edit_idx = st.number_input(
                "Instance ID",
                min_value=1,
                max_value=len(st.session_state[SS_BOXES]),
                value=1,
                step=1,
                key="edit_bbox_instance_id",
            )
        with edit_cols[1]:
            edit_bbox_str = st.text_input(
                "New BBox (x1,y1,x2,y2)",
                value="",
                placeholder="e.g. 100,120,480,700",
                key="edit_bbox_value",
            )
        with edit_cols[2]:
            st.markdown("<br>", unsafe_allow_html=True)
            apply_edit = st.button("Apply BBox", type="secondary", width="stretch")

        if apply_edit:
            raw_parts = [p.strip() for p in edit_bbox_str.replace(" ", "").split(",")]
            if len(raw_parts) != 4:
                st.error("Please input exactly 4 integers: x1,y1,x2,y2.")
            else:
                try:
                    x1, y1, x2, y2 = [int(v) for v in raw_parts]
                    x1 = max(0, min(int(settings.gen_W), x1))
                    y1 = max(0, min(int(settings.gen_H), y1))
                    x2 = max(0, min(int(settings.gen_W), x2))
                    y2 = max(0, min(int(settings.gen_H), y2))
                    if x2 <= x1 or y2 <= y1:
                        st.error("Invalid bbox after clipping: require x2>x1 and y2>y1.")
                    else:
                        target = int(edit_idx) - 1
                        st.session_state[SS_BOXES][target]["bbox"] = [x1, y1, x2, y2]
                        st.session_state[SS_ADDED_SET] = {
                            tuple(b["bbox"]) for b in st.session_state[SS_BOXES]
                        }
                        st.session_state[SS_CANVAS_KEY] += 1
                        st.success(
                            f"Updated instance {int(edit_idx)} bbox to {[x1, y1, x2, y2]}."
                        )
                        st.rerun()
                except ValueError:
                    st.error("BBox values must be integers.")

    # Editable table
    st.subheader("✅ Added instance ")
    st.markdown(
        "<p style='font-size:16px; color:gray;'>(caption is required; category is optional, then press Save Layout)</p>",
        unsafe_allow_html=True
    )
    grid = None
    if len(st.session_state[SS_BOXES]) == 0:
        st.info("No Instance. Pipeline: Draw a box first ➜ click Add Instance ➜ enter the instance caption (category is optional)")
    else:
        rows = []
        for i, a in enumerate(st.session_state[SS_BOXES]):
            x1, y1, x2, y2 = a["bbox"]
            rows.append({
                "id": i + 1,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "category": a.get("category", ""),
                "caption": a.get("caption", ""),
                "occludes": a.get("occludes", ""),
                "color": a.get("color", next_color_for(i)),
            })
        df = pd.DataFrame(rows, columns=["id", "x1", "y1", "x2", "y2", "category", "caption", "occludes", "color"])

        gob = GridOptionsBuilder.from_dataframe(df)
        gob.configure_default_column(editable=False, resizable=True, sortable=False, filter=False)
        for col in ["id", "x1", "y1", "x2", "y2", "color"]:
            gob.configure_column(col, editable=False, width=50)
        gob.configure_column("category", header_name="category", editable=True, width=120, cellEditor="agTextCellEditor")
        gob.configure_column("caption", header_name="caption", editable=True, minwidth=120, cellEditor="agTextCellEditor")
        gob.configure_column("occludes", header_name="occludes (IDs)", editable=True, width=100, cellEditor="agTextCellEditor")

        row_style_js = JsCode(
            """
function(params) {
  const c = params.data.color || '#000000';
  return { color: c };
}
"""
        )
        gob.configure_grid_options(getRowStyle=row_style_js)

        color_cell_style_js = JsCode(
            """
function(params) {
  const c = params.value || '#000000';
  function yiq(hex){
    const r=parseInt(hex.substr(1,2),16),
          g=parseInt(hex.substr(3,2),16),
          b=parseInt(hex.substr(5,2),16);
    const y=(r*299+g*587+b*114)/1000;
    return y >= 128 ? '#000000' : '#ffffff';
  }
  return { backgroundColor: c, color: yiq(c), textAlign: 'center', fontWeight: '600' };
}
"""
        )
        gob.configure_column("color", header_name="color", width=70, cellStyle=color_cell_style_js)

        gob.configure_grid_options(
            singleClickEdit=True,
            suppressClickEdit=False,
            stopEditingWhenCellsLoseFocus=True,
            enterMovesDown=True,
            enterMovesDownAfterEdit=True,
        )
        gob.configure_selection("none")
        gob.configure_grid_options(suppressDragLeaveHidesColumns=True)
        grid_options = gob.build()

        grid = AgGrid(
            df,
            gridOptions=grid_options,
            update_mode=GridUpdateMode.VALUE_CHANGED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            height=300,
            theme="streamlit",
        )

    def _collect_boxes_from_grid_or_state() -> List[Dict]:
        if grid is not None:
            edited_df = pd.DataFrame(grid["data"])
            return [
                {
                    "bbox": [int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])],
                    "category": (str(r["category"]) if pd.notna(r["category"]) and str(r["category"]).strip().lower() != "none" else "").strip(),
                    "caption": (str(r["caption"]) if pd.notna(r["caption"]) else "").strip(),
                    "occludes": (str(r["occludes"]) if pd.notna(r["occludes"]) else "").strip(),
                    "color": str(r["color"]) if pd.notna(r["color"]) else PALETTE[0],
                }
                for _, r in edited_df.iterrows()
            ]
        return list(st.session_state[SS_BOXES])

        
    cols_btn = st.columns(3)
    with cols_btn[0]:
        if st.button("Save Layout", type="secondary", width="stretch"):
            st.session_state[SS_BOXES] = _collect_boxes_from_grid_or_state()
            st.session_state[SS_AUTO_SAVE_LAYOUT] = False

    if st.session_state.get(SS_AUTO_SAVE_LAYOUT, False):
        st.session_state[SS_BOXES] = _collect_boxes_from_grid_or_state()
        st.session_state[SS_AUTO_SAVE_LAYOUT] = False

    layout_json = None
    export_boxes = _collect_boxes_from_grid_or_state()
    if len(export_boxes) > 0:
        layout_json = {
            "prompt": st.session_state[SS_PROMPT],
            "caption": st.session_state[SS_PROMPT],
            "height": settings.gen_H,
            "width": settings.gen_W,
            "annos": [
                {
                    "bbox": b["bbox"],
                    "caption": b.get("caption", ""),
                    "category_name": b.get("category", ""),
                    "occludes": b.get("occludes", ""),
                }
                for b in export_boxes
            ],
            "seed": settings.seed
        }

    with cols_btn[1]:
        if layout_json is not None:
            layout_str = json.dumps(layout_json, indent=2, ensure_ascii=False)
            st.download_button(
                "Download Layout JSON",
                data=layout_str,
                file_name="layout.json",
                mime="application/json",
                width='stretch'
            )
    
    with cols_btn[2]:
        if layout_json is not None and st.button("Save To Examples", type="secondary", width="stretch"):
            example_save_name = st.session_state.get("save_example_name", "")
            name = example_save_name.strip()
            if not name:
                st.error("Please input an example name before saving.")
            else:
                if not name.endswith(".json"):
                    name = f"{name}.json"
                examples_dir = Path("./examples")
                examples_dir.mkdir(parents=True, exist_ok=True)
                save_path = examples_dir / name
                existed = save_path.exists()
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(layout_json, f, indent=2, ensure_ascii=False)
                if existed:
                    st.success(f"Overwritten: {save_path}")
                else:
                    st.success(f"Saved to {save_path}")
                st.rerun()
        st.text_input(
            "Example Name",
            value="",
            placeholder="e.g. living_room_scene",
            key="save_example_name",
        )

    return prompt


def validate_boxes(boxes: List[Dict]) -> List[int]:
    """Return 1-based row indices that are invalid (missing caption)."""
    invalid_rows = []
    for i, b in enumerate(boxes):
        if not b.get("caption"):
            invalid_rows.append(i + 1)
    return invalid_rows


def parse_occludes_to_occluder(boxes: List[Dict]) -> List[List[int]]:
    """Convert each object's 'occludes IDs' into a reverse occluder list.
    
    Args:
        boxes: Each box contains an 'occludes' field, e.g. "1,3" means it occludes object 1 and 3.
    
    Returns:
        occluder: List[List[int]], where occluder[i] stores object indices that occlude object i.
    """
    num_objs = len(boxes)
    occluder = [[] for _ in range(num_objs)]
    
    for i, box in enumerate(boxes):
        occludes_str = box.get("occludes", "").strip()
        if occludes_str:
            # Parse comma-separated IDs (1-based).
            try:
                occluded_ids = [int(x.strip()) for x in occludes_str.split(",") if x.strip()]
                for occluded_id in occluded_ids:
                    # Convert to 0-based index.
                    occluded_idx = occluded_id - 1
                    if 0 <= occluded_idx < num_objs:
                        occluder[occluded_idx].append(i)
            except ValueError:
                pass  # Ignore parse errors.
    
    return occluder

def generate_bbox_masks(boxes: List[Dict], H: int, W: int) -> torch.Tensor:
    """Generate a binary mask for each bbox.
    
    Args:
        boxes: List of bboxes.
        H, W: Image height and width.
    
    Returns:
        bbox_masks: [num_objs, H, W] tensor
    """
    num_objs = len(boxes)
    bbox_masks = torch.zeros((num_objs, H, W), dtype=torch.float32)
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        bbox_masks[i, y1:y2, x1:x2] = 1.0
    
    return bbox_masks

def build_layout_from_boxes(boxes: List[Dict], H: int, W: int) -> Layout:
    anno_feed = [
        {
            "text": b["caption"],
            "category": b["category"],
            "bbox": xyxy2xywh(b["bbox"]),
            "hw": [int(H), int(W)],
        }
        for b in boxes
    ]
    layout = Layout(anno_feed, max_objs=MAX_OBJS)
    
    # Attach occlusion metadata.
    occluder = parse_occludes_to_occluder(boxes)
    bbox_masks = generate_bbox_masks(boxes, H, W)
    
    layout.occluder = occluder
    layout.bbox_masks = bbox_masks
    
    return layout


def compose_prompt(prompt: str, boxes: List[Dict]) -> str:
    caption_list = [
        str(box.get("caption", "")).strip()
        for box in boxes
        if str(box.get("caption", "")).strip()
    ]
    if caption_list:
        return f"{prompt}, {', '.join(caption_list)}"
    return prompt


def _format_hhmmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _progress_stats(step: int, total: int, start_time: float) -> Dict[str, float]:
    step = max(0, min(step, total))
    progress = (step / total) if total > 0 else 0.0

    elapsed = max(1e-6, time.time() - start_time)
    rate = step / elapsed if step > 0 else 0.0
    eta = ((total - step) / rate) if rate > 0 else 0.0
    return {
        "step": step,
        "total": total,
        "progress": progress,
        "elapsed": elapsed,
        "rate": rate,
        "eta": eta,
    }


def _format_tqdm_like_line(step: int, total: int, start_time: float, bar_width: int = 40) -> str:
    stats = _progress_stats(step, total, start_time)
    filled = int(round(stats["progress"] * bar_width))
    bar = "█" * filled + "░" * max(0, bar_width - filled)
    pct = int(round(stats["progress"] * 100))
    elapsed_str = _format_hhmmss(stats["elapsed"])
    eta_str = _format_hhmmss(stats["eta"])
    rate_str = f"{stats['rate']:4.2f}it/s" if stats["rate"] > 0 else " ?it/s"
    return f"{pct:3d}%|{bar}| {int(stats['step'])}/{int(stats['total'])} [{elapsed_str}<{eta_str}, {rate_str}]"


def _format_progress_html(step: int, total: int, start_time: float) -> str:
    stats = _progress_stats(step, total, start_time)
    pct = int(round(stats["progress"] * 100))
    elapsed_str = _format_hhmmss(stats["elapsed"])
    eta_str = _format_hhmmss(stats["eta"])
    rate_str = f"{stats['rate']:4.2f} it/s" if stats["rate"] > 0 else "?"
    return f"""
<div style="border:1px solid #2f3640;border-radius:10px;padding:10px 12px;background:#111827;margin:8px 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span style="color:#e5e7eb;font-weight:600;">Denoising Progress</span>
    <span style="color:#93c5fd;font-weight:600;">{pct}%</span>
  </div>
  <div style="width:100%;height:8px;background:#1f2937;border-radius:999px;overflow:hidden;">
    <div style="height:8px;width:{pct}%;background:linear-gradient(90deg,#22c55e,#38bdf8);"></div>
  </div>
  <div style="display:flex;gap:14px;margin-top:8px;color:#cbd5e1;font-size:12px;">
    <span><b>step</b> {int(stats['step'])}/{int(stats['total'])}</span>
    <span><b>elapsed</b> {elapsed_str}</span>
    <span><b>eta</b> {eta_str}</span>
    <span><b>speed</b> {rate_str}</span>
  </div>
</div>
"""


def panel_inference_and_result(settings: GenSettings, prompt: str) -> None:
    st.title("🏃 Inference")

    pipe = st.session_state.get(SS_PIPE)
    layout_transformer = st.session_state.get(SS_LT)
    is_ready = st.session_state.get(SS_LOADED, False)

    if not is_ready or pipe is None:
        st.info("Please first set the model type and path on the left, then click Load model.")
        return
    if layout_transformer is None:
        st.error("Failed to build/load (please check the ckpt path and reload from the left).")
        return

    # Validation
    can_run = (
        is_ready
        and (pipe is not None)
        and (layout_transformer is not None)
        and (len(st.session_state[SS_BOXES]) > 0)
        and (prompt != "")
    )
    if can_run:
        invalid_rows = validate_boxes(st.session_state[SS_BOXES])
        if invalid_rows:
            can_run = False
            st.error(f"Row {invalid_rows} has no caption filled in (caption is required).")

    run = st.button("🚀 Generate", type="primary", width='stretch', disabled=not can_run)
    progress_slot = st.empty()

    if not run:
        return
    total_steps = int(settings.steps)
    progress_start_time = time.time()
    progress_slot.markdown(
        _format_progress_html(0, total_steps, progress_start_time),
        unsafe_allow_html=True,
    )

    try:
        layout_obj = build_layout_from_boxes(st.session_state[SS_BOXES], settings.gen_H, settings.gen_W)

        with st.spinner("Inferencing ..."):
            g = torch.Generator(device=settings.device.type).manual_seed(int(settings.seed))
            final_prompt = compose_prompt(prompt, st.session_state[SS_BOXES]) # Necessary to align with our training process.

            def _step_callback(_pipe, i, _t, callback_kwargs):
                progress_slot.markdown(
                    _format_progress_html(i + 1, total_steps, progress_start_time),
                    unsafe_allow_html=True,
                )
                return callback_kwargs

            res = inference(
                pipeline=pipe,
                layout_transformer=layout_transformer,
                prompt=final_prompt,
                generator=g,
                num_inference_steps=int(settings.steps),
                guidance_scale=float(settings.guidance_scale),
                enable_layout=bool(settings.enable_layout),
                grounding_ratio=float(settings.grounding_ratio),
                layout=layout_obj,
                height=int(settings.gen_H),
                width=int(settings.gen_W),
                callback_on_step_end=_step_callback,
            )
            out_img: Optional[Image.Image] = res.images[0]

        if out_img is not None:
            overlay_img = layout_obj.show_layout_on_image(out_img)
            st.image(out_img, caption="Result", width='stretch')
            st.image(
                overlay_img,
                caption="Result with Layout",
                width='stretch',
            )

            ts = int(time.time())
            fn = f"gen_{settings.model_type.lower()}_{ts}.png"
            out_img_bytes = _pil_to_png_bytes(out_img)
            overlay_bytes = _pil_to_png_bytes(overlay_img)
            st.download_button(
                "Download",
                data=out_img_bytes,
                file_name=fn,
                mime="image/png",
            )
            progress_slot.markdown(
                _format_progress_html(total_steps, total_steps, progress_start_time),
                unsafe_allow_html=True,
            )

    except Exception as e:
        progress_slot.empty()
        st.exception(e)


# ---------------------- Pending Example: apply-on-start helpers ----------------------
def _apply_example_case_to_session(case: Dict):
    """Apply example data into session. MUST be called before any widget using these keys is created."""
    # prompt
    st.session_state[SS_PROMPT] = case.get("prompt", "")

    # boxes with colors
    boxes_raw = case.get("annos", [])
    new_boxes = []
    for i, b in enumerate(boxes_raw):
        col = next_color_for(i)
        new_boxes.append({
            "bbox": [int(x) for x in b["bbox"]],
            "category": (str(b.get("category", "")) if b.get("category", "") is not None and str(b.get("category", "")).strip().lower() != "none" else "").strip(),
            "caption": str(b.get("caption", "")),
            "occludes": str(b.get("occludes", "")),
            "color": col,
        })
    st.session_state[SS_BOXES] = new_boxes
    st.session_state[SS_ADDED_SET] = {tuple(b["bbox"]) for b in new_boxes}
    st.session_state[SS_SEED] = case.get("seed", st.session_state[SS_SEED])
    st.session_state[SS_HEIGHT] = case.get("height", st.session_state[SS_HEIGHT])
    st.session_state[SS_WIDTH] = case.get("width", st.session_state[SS_WIDTH])

    # bump canvas key to redraw with first color based on current len(boxes)
    st.session_state[SS_CANVAS_KEY] += 1


# ---------------------- Examples Panel (bottom) ----------------------
def _load_example_into_session(case: Dict):
    """Queue preset example into session (pending) then rerun.
    Actual write to SS_PROMPT/SS_BOXES happens at the beginning of main()
    BEFORE any widgets are instantiated.
    """
    ensure_session_state()
    st.session_state[SS_PENDING_EXAMPLE] = case   # <-- queue only
    st.session_state[SS_AUTO_SAVE_LAYOUT] = True
    st.rerun()                                    # <-- trigger rerun to apply early next run


def panel_examples() -> None:
    st.markdown("---")
    st.subheader("🧩 Examples")

    cols = st.columns(6)
    for i, case in enumerate(PRESET_CASES):
        with cols[i % 6]:
            if st.button(f"{case['name']}", key=f"example_{i}", width='stretch'):
                _load_example_into_session(case)


# ---------------------- App Entry ----------------------
def main() -> None:
    ensure_session_state()

    pending = st.session_state.get(SS_PENDING_EXAMPLE, None)
    if pending is not None:
        _apply_example_case_to_session(pending)
        st.session_state[SS_PENDING_EXAMPLE] = None
        st.toast(f"Loaded: {pending.get('name','')}", icon="✅")

    settings = sidebar_controls()
    left, right = st.columns([1.05, 1.0])
    with left:
        prompt = panel_canvas_and_table(settings)
    with right:
        panel_inference_and_result(settings, prompt)

    panel_examples()


if __name__ == "__main__":
    main()
