#!/usr/bin/env python3
# Copyright (c) 2025
# A lightweight utility to compute per-component FLOPs/params for multi-modality 3D detectors.
# It reads a config, tries to infer data shapes via the dataloader/pipeline, builds dummy inputs,
# and profiles key submodules separately. It gracefully falls back if some ops are unsupported.

from __future__ import annotations

import argparse
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
except Exception:
    Config = None  # type: ignore
    init_default_scope = None  # type: ignore
    Runner = None  # type: ignore

try:
    from mmdet3d.registry import MODELS
except Exception:
    MODELS = None  # type: ignore

# Optional profilers
try:
    from thop import profile as thop_profile  # type: ignore
    from thop import clever_format as thop_format  # type: ignore
except Exception:
    thop_profile = None  # type: ignore
    thop_format = None  # type: ignore

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count_table  # type: ignore
except Exception:
    FlopCountAnalysis = None  # type: ignore
    parameter_count_table = None  # type: ignore


def _format_number(n: Optional[float]) -> str:
    if n is None:
        return "N/A"
    if n < 0:
        return "N/A"
    units = ["", "K", "M", "G", "T", "P"]
    i = 0
    while n >= 1000 and i < len(units) - 1:
        n /= 1000.0
        i += 1
    return f"{n:.2f}{units[i]}"


def _safe_device(device: Optional[str]) -> torch.device:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _infer_img_shape_from_cfg(cfg: Config, default: Tuple[int, int] = (384, 1280)) -> Tuple[int, int]:
    # Try to parse H, W from test/val pipeline Resize/Pad
    def _scan_pipeline(pipeline: List[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
        if not isinstance(pipeline, list):
            return None
        h = w = None
        for t in pipeline:
            if not isinstance(t, dict):
                continue
            ttype = t.get("type") or t.get("_scope_")
            if ttype in ("Resize", "MResize", "MultiScaleFlipAug3D"):
                # Common keys: scale=(w, h) or img_scale
                scale = t.get("scale") or t.get("img_scale") or t.get("img_scales")
                if isinstance(scale, (list, tuple)) and len(scale) == 2:
                    w, h = int(scale[0]), int(scale[1])
                elif isinstance(scale, (list, tuple)) and len(scale) > 0 and isinstance(scale[0], (list, tuple)):
                    sw, sh = scale[0]
                    w, h = int(sw), int(sh)
            if ttype == "Pad":
                size = t.get("size")
                if isinstance(size, (list, tuple)) and len(size) == 2:
                    h, w = int(size[0]), int(size[1])
        if h and w:
            return (h, w)
        return None

    # Try multiple entries
    for key in ("test_dataloader", "val_dataloader", "train_dataloader"):
        try:
            dl = cfg.get(key)
            if dl and isinstance(dl, dict):
                dataset = dl.get("dataset")
                if isinstance(dataset, dict):
                    pipe = dataset.get("pipeline")
                    shp = _scan_pipeline(pipe)
                    if shp:
                        return shp
        except Exception:
            pass
    return default


def _infer_voxel_params_from_cfg(cfg: Config) -> Dict[str, Any]:
    # Reasonable defaults for KITTI-style configs
    params = {
        "point_cloud_range": [0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
        "voxel_size": [0.1, 0.1, 0.2],
        "max_num_points": 32,
        "max_voxels": 20000,
        "point_dims": 4,
    }

    # Look into model.pts_voxel_encoder or pipeline Voxelization
    try:
        model_cfg = cfg.get("model", {})
        # Some configs put a voxel_layer dict in the model
        for k in ("pts_voxel_layer", "voxel_layer", "voxel_encoder", "pts_voxel_encoder"):
            node = model_cfg.get(k)
            if isinstance(node, dict):
                for kk in ("point_cloud_range", "voxel_size", "max_num_points", "max_voxels"):
                    if kk in node:
                        params[kk] = node[kk]
        # Pipelines may also declare voxel params
        for key in ("test_dataloader", "val_dataloader", "train_dataloader"):
            dl = cfg.get(key)
            dataset = isinstance(dl, dict) and dl.get("dataset") or None
            pipe = isinstance(dataset, dict) and dataset.get("pipeline") or None
            if isinstance(pipe, list):
                for t in pipe:
                    if isinstance(t, dict) and t.get("type", "").lower().startswith("voxel"):
                        for kk in ("point_cloud_range", "voxel_size", "max_num_points", "max_voxels"):
                            if kk in t:
                                params[kk] = t[kk]
    except Exception:
        pass

    # Normalize types
    params["point_cloud_range"] = list(map(float, params["point_cloud_range"]))
    params["voxel_size"] = list(map(float, params["voxel_size"]))
    params["max_num_points"] = int(params["max_num_points"])
    params["max_voxels"] = int(params["max_voxels"])
    return params

def _try_build_dataloader(cfg: Config, which: str = "test"):
    """Try to build a dataloader (test/val/train). Returns dataloader or None."""
    if Runner is None:
        return None
    key = f"{which}_dataloader"
    if key not in cfg:
        # fallback order
        for alt in ("val_dataloader", "test_dataloader", "train_dataloader"):
            if alt in cfg:
                key = alt
                break
        else:
            return None
    try:
        dl = Runner.build_dataloader(cfg[key])
        return dl
    except Exception:
        return None

def _extract_shapes_from_batch(batch: Any) -> Dict[str, Any]:
    """Extract modality shapes from a collated batch produced by MMEngine dataloader.

    Returns a dict possibly containing:
    - img: dict(batch=int, c=int, h=int, w=int)
    - points: dict(num_points=int, point_dims=int)
    - voxels: dict(M=int, T=int, C=int, has_coors=bool)
    """
    shapes: Dict[str, Any] = {}
    if not isinstance(batch, dict):
        return shapes
    inputs = batch.get("inputs", batch)

    # Images
    try:
        imgs = None
        if isinstance(inputs, dict):
            imgs = inputs.get("img", None)
        elif isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict):
            imgs = inputs[0].get("img", None)
        if imgs is None and isinstance(batch, dict):
            imgs = batch.get("img", None)
        if imgs is not None:
            # imgs may be Tensor (B,C,H,W) or list of (C,H,W)
            if isinstance(imgs, torch.Tensor) and imgs.dim() == 4:
                b, c, h, w = imgs.shape
                shapes["img"] = {"batch": int(b), "c": int(c), "h": int(h), "w": int(w)}
            elif isinstance(imgs, (list, tuple)) and len(imgs) > 0 and isinstance(imgs[0], torch.Tensor):
                c, h, w = imgs[0].shape[-3:]
                shapes["img"] = {"batch": len(imgs), "c": int(c), "h": int(h), "w": int(w)}
    except Exception:
        pass

    # Points
    try:
        pts = None
        if isinstance(inputs, dict):
            pts = inputs.get("points", None)
        elif isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], dict):
            pts = inputs[0].get("points", None)
        if pts is None and isinstance(batch, dict):
            pts = batch.get("points", None)
        if pts is not None:
            # pts is usually a list of N_i x C tensors
            first = pts[0] if isinstance(pts, (list, tuple)) else pts
            if isinstance(first, torch.Tensor) and first.dim() == 2:
                n, c = first.shape
                shapes["points"] = {"num_points": int(n), "point_dims": int(c)}
    except Exception:
        pass

    # Voxels (if pipeline pre-voxelizes)
    try:
        vx = None
        if isinstance(inputs, dict):
            vx = inputs.get("voxels", None)
        if vx is None and isinstance(batch, dict):
            vx = batch.get("voxels", None)
        if isinstance(vx, dict):
            vox = vx.get("voxels", None)
            coors = vx.get("coors", None)
            nump = vx.get("num_points", None)
            if isinstance(vox, torch.Tensor) and vox.dim() == 3:
                shapes["voxels"] = {
                    "M": int(vox.shape[0]),
                    "T": int(vox.shape[1]),
                    "C": int(vox.shape[2]),
                    "has_coors": isinstance(coors, torch.Tensor) and coors.dim() == 2,
                    "has_num_points": isinstance(nump, torch.Tensor) and nump.dim() == 1,
                }
    except Exception:
        pass

    return shapes


def _build_dummy_image(batch: int, c: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    # Use uint8 -> float normalization like typical preprocessors; but it's fine to use float here
    img = torch.randn(batch, c, h, w, device=device)
    return img


def _build_dummy_points(num_points: int, point_dims: int, pcr: Sequence[float], device: torch.device) -> torch.Tensor:
    # N x C where C>=3 (x,y,z[,intensity,...])
    pts = torch.zeros((num_points, point_dims), device=device)
    x0, y0, z0, x1, y1, z1 = pcr
    pts[:, 0] = torch.rand(num_points, device=device) * (x1 - x0) + x0
    pts[:, 1] = torch.rand(num_points, device=device) * (y1 - y0) + y0
    pts[:, 2] = torch.rand(num_points, device=device) * (z1 - z0) + z0
    if point_dims > 3:
        pts[:, 3:] = torch.rand(num_points, point_dims - 3, device=device)
    return pts


def _build_dummy_voxels(voxel_params: Dict[str, Any], batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Create synthetic voxels, coors, num_points matching common HardVoxel format
    vx, vy, vz = voxel_params["voxel_size"]
    x0, y0, z0, x1, y1, z1 = voxel_params["point_cloud_range"]
    # Grid sizes
    grid_x = max(1, int(round((x1 - x0) / vx)))
    grid_y = max(1, int(round((y1 - y0) / vy)))
    grid_z = max(1, int(round((z1 - z0) / vz)))

    M = min( max(1, voxel_params.get("max_voxels", 20000)), grid_x * grid_y * max(1, min(grid_z, 10)) )
    T = max(1, voxel_params.get("max_num_points", 32))
    C = max(4, voxel_params.get("point_dims", 4))

    voxels = torch.randn((M, T, C), device=device)
    # Coors: [M, 4] -> [batch_idx, z, y, x]
    coors = torch.empty((M, 4), dtype=torch.int32, device=device)
    coors[:, 0] = 0  # single-sample batch
    coors[:, 1] = torch.randint(low=0, high=max(1, grid_z), size=(M,), device=device)
    coors[:, 2] = torch.randint(low=0, high=max(1, grid_y), size=(M,), device=device)
    coors[:, 3] = torch.randint(low=0, high=max(1, grid_x), size=(M,), device=device)

    num_points = torch.randint(low=1, high=T + 1, size=(M,), dtype=torch.int32, device=device)
    return voxels, coors, num_points


def _params_of(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _profile_with_thop(module: nn.Module, inputs: Union[torch.Tensor, Tuple, List]) -> Optional[Tuple[int, int]]:
    if thop_profile is None:
        return None
    try:
        # Case A: module expects one positional arg that is a tuple/list (e.g., neck(inputs))
        if isinstance(inputs, (tuple, list)):
            try:
                flops, params = thop_profile(module, inputs=(inputs,), verbose=False)
                return int(flops), int(params)
            except Exception:
                pass
            # Case B: module expects multiple positional args (e.g., head(x1, x2, ...))
            try:
                flops, params = thop_profile(module, inputs=tuple(inputs), verbose=False)
                return int(flops), int(params)
            except Exception:
                pass
        else:
            flops, params = thop_profile(module, inputs=(inputs,), verbose=False)
            return int(flops), int(params)
    except Exception:
        return None


def _profile_with_fvcore(module: nn.Module, inputs: Union[torch.Tensor, Tuple, List]) -> Optional[Tuple[int, int]]:
    if FlopCountAnalysis is None:
        return None
    try:
        # Try wrapped then unpacked similar to THOP
        if isinstance(inputs, (tuple, list)):
            try:
                flops = FlopCountAnalysis(module, (inputs,)).total()
                params = _params_of(module)
                return int(flops), int(params)
            except Exception:
                pass
            try:
                flops = FlopCountAnalysis(module, tuple(inputs)).total()
                params = _params_of(module)
                return int(flops), int(params)
            except Exception:
                pass
        else:
            flops = FlopCountAnalysis(module, (inputs,)).total()
        params = _params_of(module)
        return int(flops), int(params)
    except Exception:
        return None


def _profile_module(module: nn.Module, inputs: Union[torch.Tensor, Tuple, List]) -> Dict[str, Any]:
    # Try THOP, then fvcore, else return params only
    out: Dict[str, Any] = {"flops": None, "params": _params_of(module)}
    res = _profile_with_thop(module, inputs)
    if res is None:
        res = _profile_with_fvcore(module, inputs)
    if res is not None:
        out["flops"], out["params"] = res
    return out


def _estimate_voxel_encoder_flops(module: nn.Module, voxels: torch.Tensor, num_points: torch.Tensor) -> Optional[int]:
    """Estimate FLOPs for voxel encoder (typically PointNet-like structure).
    
    Args:
        module: The voxel encoder module
        voxels: Voxel features [M, T, C] where M=num_voxels, T=max_points, C=point_dims
        num_points: Actual number of points per voxel [M]
    
    Returns:
        Estimated FLOPs or None if calculation fails
    """
    try:
        M, T, C = voxels.shape
        total_flops = 0
        
        # Count actual points across all voxels
        actual_points = num_points.sum().item()
        
        # Recursively count all Linear/Conv layers
        def count_layers(m, points):
            flops = 0
            for child in m.children():
                if isinstance(child, nn.Linear):
                    # FLOPs = 2 * in * out * num_points (MACs counted as 2 ops)
                    flops += 2 * child.in_features * child.out_features * points
                elif isinstance(child, nn.Conv1d):
                    in_ch = child.in_channels
                    out_ch = child.out_channels
                    kernel = child.kernel_size[0] if isinstance(child.kernel_size, tuple) else child.kernel_size
                    # Approximate for per-point convolution
                    flops += 2 * in_ch * out_ch * kernel * points
                elif isinstance(child, nn.Conv2d):
                    # For DynamicVFE which might use Conv2d
                    in_ch = child.in_channels
                    out_ch = child.out_channels
                    k_h, k_w = child.kernel_size if isinstance(child.kernel_size, tuple) else (child.kernel_size, child.kernel_size)
                    # Approximate assuming point-wise operation
                    flops += 2 * k_h * k_w * in_ch * out_ch * points
                else:
                    # Recursively process nested modules
                    flops += count_layers(child, points)
            return flops
        
        total_flops = count_layers(module, actual_points)
        
        return int(total_flops) if total_flops > 0 else None
    except Exception as e:
        # Debug: uncomment to see errors
        # print(f"Voxel encoder FLOPs estimation failed: {e}")
        return None


def _estimate_sparse_middle_encoder_flops(module: nn.Module, voxel_shape: Tuple[int, ...], 
                                          num_active_voxels: int) -> Optional[int]:
    """Estimate FLOPs for sparse middle encoder (3D sparse convolutions).
    
    Args:
        module: The middle encoder module
        voxel_shape: Shape of voxel grid (typically [C, Z, Y, X])
        num_active_voxels: Number of non-empty voxels
    
    Returns:
        Estimated FLOPs or None
    """
    try:
        total_flops = 0
        
        # For sparse convolutions, FLOPs proportional to active voxels, not full grid
        # Typical sparse conv: ~O(num_active_voxels * kernel_volume * in_ch * out_ch)
        
        # Scan for sparse conv blocks
        def count_sparse_conv_layers(m, active_sites):
            flops = 0
            for child in m.children():
                # Check for common sparse conv module names/types
                child_type = type(child).__name__
                if 'SparseConv' in child_type or 'SubMConv' in child_type:
                    # Estimate: kernel_vol * in_ch * out_ch * active_sites * 2 (MAC)
                    if hasattr(child, 'in_channels') and hasattr(child, 'out_channels'):
                        in_ch = child.in_channels
                        out_ch = child.out_channels
                        # Typical 3x3x3 kernel
                        kernel_vol = 27
                        if hasattr(child, 'kernel_size'):
                            k = child.kernel_size
                            kernel_vol = (k[0] if isinstance(k, (list, tuple)) else k) ** 3
                        flops += 2 * in_ch * out_ch * kernel_vol * active_sites
                        # Update active sites (usually stays similar in sparse conv)
                        active_sites = int(active_sites * 1.1)  # Slight increase per layer
                elif hasattr(child, 'children'):
                    flops += count_sparse_conv_layers(child, active_sites)
            return flops
        
        total_flops = count_sparse_conv_layers(module, num_active_voxels)
        return int(total_flops) if total_flops > 0 else None
    except Exception:
        return None


def _estimate_conv_backbone_flops(module: nn.Module, input_tensor: torch.Tensor) -> Optional[int]:
    """Fallback FLOPs estimator for standard CNN backbone.
    
    Args:
        module: Backbone module (typically ResNet-like)
        input_tensor: Input tensor [B, C, H, W]
    
    Returns:
        Estimated FLOPs or None
    """
    try:
        B, C, H, W = input_tensor.shape
        total_flops = 0
        
        def count_conv_flops(m, h, w, b):
            flops = 0
            out_h, out_w = h, w
            for child in m.children():
                if isinstance(child, nn.Conv2d):
                    in_ch = child.in_channels
                    out_ch = child.out_channels
                    k_h, k_w = child.kernel_size if isinstance(child.kernel_size, tuple) else (child.kernel_size, child.kernel_size)
                    stride_h, stride_w = child.stride if isinstance(child.stride, tuple) else (child.stride, child.stride)
                    pad_h = child.padding[0] if isinstance(child.padding, tuple) else child.padding
                    pad_w = child.padding[1] if isinstance(child.padding, tuple) else child.padding
                    out_h = (h + 2 * pad_h - k_h) // stride_h + 1
                    out_w = (w + 2 * pad_w - k_w) // stride_w + 1
                    # FLOPs = 2 * K_h * K_w * C_in * C_out * H_out * W_out * B
                    flops += 2 * k_h * k_w * in_ch * out_ch * out_h * out_w * b
                    h, w = out_h, out_w
                elif isinstance(child, nn.Linear):
                    # Handle linear layers (flatten assumed)
                    in_features = child.in_features
                    out_features = child.out_features
                    flops += 2 * in_features * out_features * b
                else:
                    # Recursively process nested modules
                    sub_flops, h, w = count_conv_flops(child, h, w, b)
                    flops += sub_flops
                    out_h, out_w = h, w
            return flops, out_h, out_w
        
        total_flops, _, _ = count_conv_flops(module, H, W, B)
        return int(total_flops) if total_flops > 0 else None
    except Exception as e:
        # Debug: uncomment to see errors
        # print(f"Backbone FLOPs estimation failed: {e}")
        return None


def compute_multimodal_component_flops(
    config_path: str,
    device: Optional[str] = None,
    batch_img: int = 1,
    img_c: int = 3,
    img_hw: Optional[Tuple[int, int]] = None,
    num_points: int = 40000,
    point_dims: int = 4,
    use_dataloader: bool = True,
    dataloader_split: str = "test"
) -> Dict[str, Dict[str, Any]]:
    """Compute per-component FLOPs/params for a multi-modality detector.

    Returns a dict keyed by component names with fields: flops, params.
    Some components may report params only if FLOPs unsupported.
    """
    if Config is None or MODELS is None:
        raise RuntimeError("MMEngine/MMDet3D not available in environment.")

    dev = _safe_device(device)
    cfg = Config.fromfile(config_path)
    init_default_scope("mmdet3d")

    # Build model
    model: nn.Module = MODELS.build(cfg.model)  # type: ignore
    model.to(dev)
    model.eval()

    # Infer shapes
    if img_hw is None:
        H, W = _infer_img_shape_from_cfg(cfg)
    else:
        H, W = img_hw

    voxel_params = _infer_voxel_params_from_cfg(cfg)
    voxel_params["point_dims"] = point_dims

    # Optionally sample a real batch to override inferred shapes
    if use_dataloader:
        dl = _try_build_dataloader(cfg, which=dataloader_split)
        if dl is not None:
            try:
                batch = next(iter(dl))
                shapes = _extract_shapes_from_batch(batch)
                # Update image shape
                if "img" in shapes:
                    H, W = int(shapes["img"]["h"]), int(shapes["img"]["w"])
                    batch_img = int(shapes["img"]["batch"]) or batch_img
                    img_c = int(shapes["img"]["c"]) or img_c
                # Update point shapes
                if "points" in shapes:
                    num_points = int(shapes["points"]["num_points"]) or num_points
                    point_dims = int(shapes["points"]["point_dims"]) or point_dims
                    voxel_params["point_dims"] = point_dims
                # If voxels provided, prefer those sizes for synthetic generation bounds
                if "voxels" in shapes:
                    voxel_params["max_voxels"] = max(voxel_params["max_voxels"], int(shapes["voxels"]["M"]))
                    voxel_params["max_num_points"] = max(voxel_params["max_num_points"], int(shapes["voxels"]["T"]))
            except Exception:
                pass

    # Build dummies
    img = _build_dummy_image(batch_img, img_c, H, W, dev)
    points = _build_dummy_points(num_points, point_dims, voxel_params["point_cloud_range"], dev)
    voxels, coors, nump = _build_dummy_voxels(voxel_params, batch_img, dev)

    report: Dict[str, Dict[str, Any]] = {}

    # Image backbone
    if hasattr(model, "img_backbone") and isinstance(model.img_backbone, nn.Module):  # type: ignore
        res = _profile_module(model.img_backbone, img)
        report["img_backbone"] = {"flops": _format_number(res.get("flops")), "params": _format_number(res.get("params"))}
        try:
            with torch.no_grad():
                img_feats = model.img_backbone(img)
        except Exception:
            img_feats = None
    else:
        img_feats = None

    # Image neck (if any)
    if hasattr(model, "img_neck") and isinstance(model.img_neck, nn.Module) and img_feats is not None:  # type: ignore
        # Ensure input is a tuple/list as most necks expect a sequence of feature maps
        feats = img_feats if isinstance(img_feats, (tuple, list)) else (img_feats,)
        res = _profile_module(model.img_neck, feats)
        report["img_neck"] = {"flops": _format_number(res.get("flops")), "params": _format_number(res.get("params"))}
        try:
            with torch.no_grad():
                img_feats_out = model.img_neck(*feats) if not isinstance(feats, (tuple, list)) else model.img_neck(feats)  # type: ignore
        except Exception:
            img_feats_out = None
    else:
        img_feats_out = None

    # Points voxel encoder
    if hasattr(model, "pts_voxel_encoder") and isinstance(model.pts_voxel_encoder, nn.Module):  # type: ignore
        # Common signature: (voxels, coors, num_points)
        # Some encoders accept dict; try both
        enc_inputs: Union[Tuple[Any, ...], Dict[str, Any]] = (voxels, coors, nump)
        try:
            res = _profile_module(model.pts_voxel_encoder, enc_inputs)  # type: ignore
            # If standard profiling failed, try custom estimator
            if res.get("flops") is None:
                custom_flops = _estimate_voxel_encoder_flops(model.pts_voxel_encoder, voxels, nump)  # type: ignore
                if custom_flops is not None:
                    res["flops"] = custom_flops
            report["pts_voxel_encoder"] = {"flops": _format_number(res.get("flops")), "params": _format_number(res.get("params"))}
            with torch.no_grad():
                voxel_feats = model.pts_voxel_encoder(voxels, coors, nump)  # type: ignore
        except Exception:
            # Fallback: try custom estimator, then params only
            custom_flops = _estimate_voxel_encoder_flops(model.pts_voxel_encoder, voxels, nump)  # type: ignore
            params = _params_of(model.pts_voxel_encoder)  # type: ignore
            report["pts_voxel_encoder"] = {"flops": _format_number(custom_flops), "params": _format_number(params)}
            voxel_feats = None
    else:
        voxel_feats = None

    # Points middle encoder (sparse conv)
    if hasattr(model, "pts_middle_encoder") and isinstance(model.pts_middle_encoder, nn.Module):  # type: ignore
        # Many middle encoders expect a spconv SparseTensor, which we don't synthesize here.
        # Use custom FLOPs estimator for sparse convolutions
        try:
            if voxel_feats is not None:
                with torch.no_grad():
                    mid_out = model.pts_middle_encoder(voxel_feats, coors, batch_img)  # type: ignore[attr-defined]
            else:
                mid_out = None
        except Exception:
            mid_out = None
        
        # Try to estimate FLOPs for sparse middle encoder
        M = voxels.shape[0]  # Number of active voxels
        voxel_shape = (128, 10, 400, 352)  # Typical BEV grid shape
        custom_flops = _estimate_sparse_middle_encoder_flops(model.pts_middle_encoder, voxel_shape, M)  # type: ignore
        params = _params_of(model.pts_middle_encoder)  # type: ignore
        report["pts_middle_encoder"] = {"flops": _format_number(custom_flops), "params": _format_number(params)}
    else:
        mid_out = None

    # Points backbone (BEV backbone)
    if hasattr(model, "pts_backbone") and isinstance(model.pts_backbone, nn.Module):  # type: ignore
        # Expecting a 4D BEV feature map: (B, C, H, W). Try to derive from mid_out, else guess.
        bev = None
        if isinstance(mid_out, torch.Tensor) and mid_out.dim() == 4:
            bev = mid_out
        else:
            # Guess typical BEV shape
            C = 128
            H_bev, W_bev = 200, 176
            bev = torch.randn((batch_img, C, H_bev, W_bev), device=dev)
        res = _profile_module(model.pts_backbone, bev)  # type: ignore
        # If standard profiling failed, try custom estimator
        if res.get("flops") is None:
            custom_flops = _estimate_conv_backbone_flops(model.pts_backbone, bev)  # type: ignore
            if custom_flops is not None:
                res["flops"] = custom_flops
        report["pts_backbone"] = {"flops": _format_number(res.get("flops")), "params": _format_number(res.get("params"))}
        try:
            with torch.no_grad():
                pts_feats = model.pts_backbone(bev)  # type: ignore
        except Exception:
            pts_feats = None
    else:
        pts_feats = None

    # Fusion layer (if any)
    if hasattr(model, "fusion_layer") and isinstance(model.fusion_layer, nn.Module):  # type: ignore
        # Create minimal plausible inputs
        if img_feats_out is None and img_feats is not None:
            img_feats_in = img_feats
        else:
            img_feats_in = img_feats_out
        if isinstance(img_feats_in, torch.Tensor):
            img_feats_in = (img_feats_in,)
        if isinstance(pts_feats, torch.Tensor):
            pts_feats_in = (pts_feats,)
        else:
            pts_feats_in = pts_feats
        fusion_inputs = (img_feats_in, pts_feats_in)
        try:
            res = _profile_module(model.fusion_layer, fusion_inputs)  # type: ignore
            report["fusion_layer"] = {"flops": _format_number(res.get("flops")), "params": _format_number(res.get("params"))}
        except Exception:
            report["fusion_layer"] = {"flops": "N/A", "params": _format_number(_params_of(model.fusion_layer))}  # type: ignore

    # Head(s)
    if hasattr(model, "bbox_head") and isinstance(model.bbox_head, nn.Module):  # type: ignore
        # Try to provide both img and pts feats if available
        head_in: Any = None
        if pts_feats is not None and isinstance(pts_feats, (tuple, list)):
            head_in = pts_feats
        elif pts_feats is not None:
            head_in = (pts_feats,)
        elif img_feats_out is not None and isinstance(img_feats_out, (tuple, list)):
            head_in = img_feats_out
        elif img_feats_out is not None:
            head_in = (img_feats_out,)
        if head_in is not None:
            res = _profile_module(model.bbox_head, head_in)  # type: ignore
            report["bbox_head"] = {"flops": _format_number(res.get("flops")), "params": _format_number(res.get("params"))}
        else:
            report["bbox_head"] = {"flops": "N/A", "params": _format_number(_params_of(model.bbox_head))}  # type: ignore

    # Neck (BEV img neck separate from img_neck)
    if hasattr(model, "neck") and isinstance(model.neck, nn.Module):  # type: ignore
        # Try pass pts_feats or img_feats
        neck_in: Any = pts_feats if pts_feats is not None else img_feats_out or img_feats
        if neck_in is not None:
            if not isinstance(neck_in, (tuple, list)):
                neck_in = (neck_in,)
            res = _profile_module(model.neck, neck_in)  # type: ignore
            report["neck"] = {"flops": _format_number(res.get("flops")), "params": _format_number(res.get("params"))}
        else:
            report["neck"] = {"flops": "N/A", "params": _format_number(_params_of(model.neck))}  # type: ignore

    return report


def _print_report(report: Dict[str, Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print(f"{'MULTI-MODALITY COMPONENT FLOPS':^80}")
    print("=" * 80)
    for name, stats in report.items():
        print(f"- {name:20s}  FLOPs: {stats.get('flops', 'N/A'):>10s}   Params: {stats.get('params', 'N/A'):>10s}")
    print("=" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Per-component FLOPs for multi-modality models")
    parser.add_argument("config", type=str, help="Path to config file")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda:0, etc.")
    parser.add_argument("--batch-img", type=int, default=1)
    parser.add_argument("--img-shape", type=int, nargs=2, metavar=("H", "W"), default=None)
    parser.add_argument("--num-points", type=int, default=40000)
    parser.add_argument("--point-dims", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rep = compute_multimodal_component_flops(
        config_path=args.config,
        device=args.device,
        batch_img=args.batch_img,
        img_hw=tuple(args.img_shape) if args.img_shape is not None else None,
        num_points=args.num_points,
        point_dims=args.point_dims,
    )
    _print_report(rep)


if __name__ == "__main__":
    main()
