import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from diffusers.pipelines import FluxPipeline

from src.occlusionformer.inference import inference as flux_generate
from src.occlusionformer.tools import Layout
from src.occlusionformer.transformer import OcclusionFormerFluxTransformer2DModel
from src.utils import xyxy2xywh


MAX_OBJS = 50


@dataclass
class InferenceConfig:
    model_path: str
    ckpt_path: str
    output_dir: str
    layout_json: Optional[str]
    layout_dir: Optional[str]
    steps: int
    guidance_scale: float
    grounding_ratio: float
    seed: int
    enable_layout: bool
    dtype: torch.dtype
    device: torch.device
    overwrite: bool


def parse_args() -> InferenceConfig:
    parser = argparse.ArgumentParser(
        description="OcclusionFormer CLI inference (no frontend UI)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Backbone model path or model id.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Merged .safetensors checkpoint or legacy directory containing pytorch_lora_weights.safetensors and layout.pth.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs_occlusionformer",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--layout_json",
        type=str,
        default=None,
        help="Single layout json file path.",
    )
    parser.add_argument(
        "--layout_dir",
        type=str,
        default=None,
        help="Directory containing layout json files for batch inference.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--grounding_ratio",
        type=float,
        default=0.3,
        help="Layout grounding ratio in [0,1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--enable_layout",
        action="store_true",
        help="Enable layout conditioning.",
    )
    parser.add_argument(
        "--disable_layout",
        action="store_true",
        help="Disable layout conditioning.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Torch dtype for inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device string, e.g., cuda:0 or cpu.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )

    args = parser.parse_args()

    if args.layout_json is None and args.layout_dir is None:
        parser.error("At least one of --layout_json or --layout_dir is required.")
    if args.layout_json is not None and args.layout_dir is not None:
        parser.error("Use either --layout_json or --layout_dir, not both.")

    if args.enable_layout and args.disable_layout:
        parser.error("--enable_layout and --disable_layout are mutually exclusive.")

    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    enable_layout = True
    if args.disable_layout:
        enable_layout = False
    elif args.enable_layout:
        enable_layout = True

    return InferenceConfig(
        model_path=args.model_path,
        ckpt_path=args.ckpt_path,
        output_dir=args.output_dir,
        layout_json=args.layout_json,
        layout_dir=args.layout_dir,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        grounding_ratio=args.grounding_ratio,
        seed=args.seed,
        enable_layout=enable_layout,
        dtype=dtype,
        device=torch.device(args.device),
        overwrite=args.overwrite,
    )


def _build_flux_layout_transformer_from_ckpt(
    model_path: str,
    ckpt_dir: Path,
    dtype: torch.dtype,
    device: torch.device,
):
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
    FluxPipeline.load_lora_into_transformer(
        state_dict=lora_sd,
        network_alphas=alphas,
        transformer=layout_transformer,
    )

    layout_file = ckpt_dir / "occ.pth"
    if not layout_file.exists():
        raise FileNotFoundError(f"Missing layout state dict: {layout_file}")

    # Load layout weights and keep dtype consistent.
    layout_state_dict = torch.load(layout_file, map_location="cpu", weights_only=False)
    layout_state_dict = {
        k: v.to(dtype) if v.dtype.is_floating_point else v
        for k, v in layout_state_dict.items()
    }
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

    return layout_transformer.to(dtype=dtype, device=device)


def load_flux_bundle(
    model_path: str,
    ckpt_path: str,
    dtype: torch.dtype,
    device: torch.device,
):
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype).to(device)
    pipe.transformer.to("cpu")
    layout_transformer = _build_flux_layout_transformer_from_ckpt(
        model_path=model_path,
        ckpt_dir=Path(ckpt_path),
        dtype=dtype,
        device=device,
    )
    return pipe, layout_transformer


def parse_occludes_to_occluder(boxes: List[Dict]) -> List[List[int]]:
    num_objs = len(boxes)
    occluder = [[] for _ in range(num_objs)]

    for i, box in enumerate(boxes):
        occludes_str = str(box.get("occludes", "")).strip()
        if not occludes_str:
            continue
        try:
            occluded_ids = [int(x.strip()) for x in occludes_str.split(",") if x.strip()]
        except ValueError:
            continue
        for occluded_id in occluded_ids:
            occluded_idx = occluded_id - 1
            if 0 <= occluded_idx < num_objs:
                occluder[occluded_idx].append(i)

    return occluder


def generate_bbox_masks(boxes: List[Dict], height: int, width: int) -> torch.Tensor:
    num_objs = len(boxes)
    bbox_masks = torch.zeros((num_objs, height, width), dtype=torch.float32)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 > x1 and y2 > y1:
            bbox_masks[i, y1:y2, x1:x2] = 1.0
    return bbox_masks


def _normalize_bbox(bbox: List[float]) -> List[int]:
    return [int(round(float(v))) for v in bbox]


def _is_xyxy_bbox(bbox: List[int]) -> bool:
    return len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]


def _layout_boxes_from_json(annos: List[Dict]) -> List[Dict]:
    boxes = []
    for anno in annos:
        raw_bbox = anno.get("bbox", [])
        if len(raw_bbox) != 4:
            continue
        bbox = _normalize_bbox(raw_bbox)
        if not _is_xyxy_bbox(bbox):
            x, y, w, h = bbox
            bbox = [x, y, x + max(0, w), y + max(0, h)]
        boxes.append(
            {
                "bbox": bbox,
                "category": str(anno.get("category", anno.get("category_name", ""))).strip(),
                "caption": str(anno.get("caption", "")).strip(),
                "occludes": str(anno.get("occludes", "")).strip(),
            }
        )
    return boxes


def validate_boxes(boxes: List[Dict]) -> None:
    invalid_rows = []
    for i, box in enumerate(boxes):
        if (not box.get("category")) and (not box.get("caption")):
            invalid_rows.append(i + 1)
    if invalid_rows:
        raise ValueError(
            f"Invalid annos rows {invalid_rows}: both category and caption are empty."
        )


def build_layout_from_boxes(boxes: List[Dict], height: int, width: int) -> Layout:
    anno_feed = []
    for box in boxes:
        anno_feed.append(
            {
                "text": box["caption"],
                "category": box["category"],
                "bbox": xyxy2xywh(box["bbox"]),
                "hw": [int(height), int(width)],
            }
        )
    layout = Layout(anno_feed, max_objs=MAX_OBJS)
    layout.occluder = parse_occludes_to_occluder(boxes)
    layout.bbox_masks = generate_bbox_masks(boxes, height, width)
    return layout


def compose_prompt(global_prompt: str, boxes: List[Dict]) -> str:
    caption_list = [b["caption"].strip() for b in boxes if b.get("caption", "").strip()]
    if caption_list:
        return f"{global_prompt}, {', '.join(caption_list)}"
    return global_prompt


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def save_results(
    output_dir: Path,
    stem: str,
    image: Image.Image,
    overlay: Image.Image,
    overwrite: bool,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = sanitize_name(stem) or "result"
    image_path = output_dir / f"{base}.png"
    overlay_path = output_dir / f"{base}_layout.png"
    if (image_path.exists() or overlay_path.exists()) and not overwrite:
        raise FileExistsError(
            f"Output exists: {image_path} or {overlay_path}. Use --overwrite to replace."
        )
    image.save(image_path)
    overlay.save(overlay_path)
    return image_path, overlay_path


def load_layout_json_paths(layout_json: Optional[str], layout_dir: Optional[str]) -> List[Path]:
    if layout_json:
        path = Path(layout_json)
        if not path.exists():
            raise FileNotFoundError(f"Layout json not found: {path}")
        return [path]

    directory = Path(layout_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Layout directory not found: {directory}")

    paths = sorted([p for p in directory.glob("*.json") if p.is_file()], key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError(f"No json files found in directory: {directory}")
    return paths


def run_single_layout(
    path: Path,
    pipe: FluxPipeline,
    layout_transformer,
    cfg: InferenceConfig,
) -> Tuple[Path, Path]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    prompt = str(payload.get("prompt", payload.get("caption", ""))).strip()
    if not prompt:
        raise ValueError(f"Missing prompt/caption in {path}")

    height = int(payload.get("height", 1024))
    width = int(payload.get("width", 1024))
    annos = payload.get("annos", [])
    if not isinstance(annos, list) or not annos:
        raise ValueError(f"Missing or empty annos in {path}")

    boxes = _layout_boxes_from_json(annos)
    if not boxes:
        raise ValueError(f"No valid boxes in {path}")
    validate_boxes(boxes)

    layout_obj = build_layout_from_boxes(boxes, height, width)
    final_prompt = compose_prompt(prompt, boxes) # Necessary to align with our training process.

    generator = torch.Generator(device=cfg.device.type).manual_seed(cfg.seed)
    result = flux_generate(
        pipeline=pipe,
        layout_transformer=layout_transformer,
        prompt=final_prompt,
        generator=generator,
        num_inference_steps=int(cfg.steps),
        guidance_scale=float(cfg.guidance_scale),
        enable_layout=bool(cfg.enable_layout),
        grounding_ratio=float(cfg.grounding_ratio),
        layout=layout_obj,
        height=height,
        width=width,
    )
    out_img = result.images[0]
    overlay = layout_obj.show_layout_on_image(out_img)

    return save_results(
        output_dir=Path(cfg.output_dir),
        stem=path.stem,
        image=out_img,
        overlay=overlay,
        overwrite=cfg.overwrite,
    )


def main() -> None:
    cfg = parse_args()
    print(f"[INFO] device={cfg.device}, dtype={cfg.dtype}, enable_layout={cfg.enable_layout}")
    print(f"[INFO] loading model from {cfg.model_path}")
    pipe, layout_transformer = load_flux_bundle(
        model_path=cfg.model_path,
        ckpt_path=cfg.ckpt_path,
        dtype=cfg.dtype,
        device=cfg.device,
    )
    print("[INFO] model loaded")

    paths = load_layout_json_paths(cfg.layout_json, cfg.layout_dir)
    print(f"[INFO] total layout files: {len(paths)}")

    ok = 0
    failed = 0
    for i, path in enumerate(paths, start=1):
        try:
            img_path, overlay_path = run_single_layout(path, pipe, layout_transformer, cfg)
            ok += 1
            print(f"[{i}/{len(paths)}] OK {path.name} -> {img_path.name}, {overlay_path.name}")
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(paths)}] FAIL {path.name}: {exc}")

    print(f"[DONE] success={ok}, failed={failed}, output_dir={cfg.output_dir}")


if __name__ == "__main__":
    main()
