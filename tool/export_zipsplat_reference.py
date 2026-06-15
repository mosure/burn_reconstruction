#!/usr/bin/env python3
"""Export official ZipSplat upstream outputs for Burn parity tests.

This is offline correctness tooling only. Runtime inference remains Burn/Rust.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import save_file


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def install_inference_only_stubs() -> None:
    """Stub optional rendering/visualization modules unused by model forward."""

    if "gsplat" not in sys.modules:
        gsplat = types.ModuleType("gsplat")
        rendering = types.ModuleType("gsplat.rendering")

        def rasterization(*_args, **_kwargs):
            raise RuntimeError("gsplat rasterization is not needed for reference export")

        rendering.rasterization = rasterization
        gsplat.rendering = rendering
        sys.modules["gsplat"] = gsplat
        sys.modules["gsplat.rendering"] = rendering

    if "plyfile" not in sys.modules:
        plyfile = types.ModuleType("plyfile")

        class _PlyData:
            pass

        class _PlyElement:
            @staticmethod
            def describe(*_args, **_kwargs):
                raise RuntimeError("PLY export is not needed for reference export")

        plyfile.PlyData = _PlyData
        plyfile.PlyElement = _PlyElement
        sys.modules["plyfile"] = plyfile

    if "matplotlib" not in sys.modules:
        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")

        def get_cmap(*_args, **_kwargs):
            raise RuntimeError("matplotlib colormaps are not needed for reference export")

        pyplot.get_cmap = get_cmap
        matplotlib.pyplot = pyplot
        sys.modules["matplotlib"] = matplotlib
        sys.modules["matplotlib.pyplot"] = pyplot

    if "imageio" not in sys.modules:
        imageio = types.ModuleType("imageio")
        imageio_v2 = types.ModuleType("imageio.v2")
        imageio.v2 = imageio_v2
        sys.modules["imageio"] = imageio
        sys.modules["imageio.v2"] = imageio_v2


def build_covariance(scales: torch.Tensor, rotations_xyzw: torch.Tensor) -> torch.Tensor:
    flat_scales = scales.reshape(-1, 3)
    q = rotations_xyzw.reshape(-1, 4)
    i, j, k, r = q.unbind(dim=-1)
    two_s = 2.0 / (q.square().sum(dim=-1).clamp_min(1e-8))
    rows = torch.stack(
        (
            1.0 - two_s * (j.square() + k.square()),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1.0 - two_s * (i.square() + k.square()),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1.0 - two_s * (i.square() + j.square()),
        ),
        dim=-1,
    )
    rotation = rows.reshape(-1, 3, 3)
    rs = rotation * flat_scales.unsqueeze(1)
    cov = rs @ rs.transpose(1, 2)
    return cov.reshape(*scales.shape[:-1], 3, 3)


def parse_args() -> argparse.Namespace:
    root = workspace_root()
    parser = argparse.ArgumentParser(
        description="Export official ZipSplat upstream tensor outputs as safetensors."
    )
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("/tmp/ZipSplat"),
        help="Path to a checkout of https://github.com/cvg/ZipSplat.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path.home()
        / ".burn_reconstruction/models/zipsplat/zipsplat-da3g-252p.tar",
        help="Official ZipSplat PyTorch checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "assets/fixtures/zipsplat_reference.safetensors",
        help="Output safetensors fixture.",
    )
    parser.add_argument(
        "--image",
        dest="images",
        action="append",
        type=Path,
        default=[
            root / "assets/images/re10k/0.png",
            root / "assets/images/re10k/1.png",
            root / "assets/images/re10k/2.png",
        ],
        help="Input image path. May be passed multiple times.",
    )
    parser.add_argument(
        "--compression",
        type=float,
        default=1.0,
        help="Upstream ZipSplat compression ratio. 1.0 bypasses k-means.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for reference inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.upstream.exists():
        raise SystemExit(f"missing upstream ZipSplat checkout: {args.upstream}")
    if not args.weights.exists():
        raise SystemExit(f"missing official ZipSplat checkpoint: {args.weights}")
    for image in args.images:
        if not image.exists():
            raise SystemExit(f"missing input image: {image}")

    install_inference_only_stubs()
    sys.path.insert(0, str(args.upstream))

    from zipsplat.predictor import ZipSplat
    from zipsplat.utils import load_image

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    device = torch.device(args.device)
    model = ZipSplat(weights=str(args.weights)).to(device).eval()
    raw_images = [load_image(image) for image in args.images]
    images, _cameras, _poses = model._prepare_inputs(raw_images, None, None)

    with torch.no_grad():
        gaussians = model.model(images, compression=args.compression)

    quats_wxyz = gaussians.quats.float().cpu()
    quats_xyzw = torch.cat([quats_wxyz[..., 1:], quats_wxyz[..., :1]], dim=-1).contiguous()
    scales = gaussians.scales.float().cpu()
    tensors = {
        "input_images": images.float().cpu().contiguous(),
        "gaussians_means": gaussians.means.float().cpu().contiguous(),
        "gaussians_covariances": build_covariance(scales, quats_xyzw).float().cpu().contiguous(),
        "gaussians_harmonics": gaussians.sh_coeffs.float()
        .cpu()
        .permute(0, 1, 3, 2)
        .contiguous(),
        "gaussians_opacities": gaussians.opacities.float().cpu().contiguous(),
        "gaussians_rotations": quats_xyzw,
        "gaussians_rotations_wxyz": quats_wxyz.contiguous(),
        "gaussians_scales": scales.contiguous(),
    }
    metadata = {
        "source": "https://github.com/cvg/ZipSplat",
        "upstream_path": str(args.upstream),
        "weights": str(args.weights),
        "images": json.dumps([str(path) for path in args.images]),
        "compression": str(args.compression),
        "device": str(device),
        "torch": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=metadata)
    print(f"wrote {args.output}")
    for name, tensor in tensors.items():
        print(f"{name}: shape={tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
