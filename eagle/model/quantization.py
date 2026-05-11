"""Shared quantization options for LTD training and evaluation."""

from __future__ import annotations

from typing import Dict, Optional

import torch


VALID_QUANTIZATION_MODES = ("fp16", "bf16", "int8", "int4")


def add_quantization_args(parser, default: str = "fp16") -> None:
    parser.add_argument(
        "--quantization",
        choices=VALID_QUANTIZATION_MODES,
        default=default,
        help=(
            "Base/EAGLE model precision for the environment or benchmark. "
            "int8/int4 use bitsandbytes quantized loading."
        ),
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Deprecated alias for --quantization int8.",
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        choices=("fp16", "bf16"),
        default="bf16",
        help="Compute dtype used by bitsandbytes 4-bit layers.",
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        choices=("fp4", "nf4"),
        default="nf4",
        help="4-bit quantization type used by bitsandbytes.",
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        dest="bnb_4bit_use_double_quant",
        action="store_true",
        default=True,
        help="Enable nested quantization for bitsandbytes 4-bit loading.",
    )
    parser.add_argument(
        "--no_bnb_4bit_use_double_quant",
        dest="bnb_4bit_use_double_quant",
        action="store_false",
        help="Disable nested quantization for bitsandbytes 4-bit loading.",
    )


def resolve_quantization(args) -> str:
    mode = getattr(args, "quantization", "fp16").lower()
    if mode not in VALID_QUANTIZATION_MODES:
        raise ValueError(
            f"Unknown quantization mode {mode!r}; expected one of {VALID_QUANTIZATION_MODES}."
        )

    if getattr(args, "load_in_8bit", False):
        if mode not in ("fp16", "int8"):
            raise ValueError("--load_in_8bit cannot be combined with --quantization other than int8.")
        mode = "int8"

    return mode


def dtype_from_name(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype {dtype_name!r}")


def model_torch_dtype(mode: str) -> torch.dtype:
    if mode == "bf16":
        return torch.bfloat16
    return torch.float16


def effective_device_map(mode: str, device_map: str) -> str:
    if device_map:
        return device_map
    if mode in ("int8", "int4"):
        return "auto"
    return ""


def apply_quantization_config(model_kwargs: Dict, args) -> str:
    mode = resolve_quantization(args)
    model_kwargs["torch_dtype"] = model_torch_dtype(mode)

    device_map = effective_device_map(mode, getattr(args, "device_map", ""))
    if mode == "int8":
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = device_map
    elif mode == "int4":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "INT4 loading requires transformers BitsAndBytesConfig and bitsandbytes."
            ) from exc

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype_from_name(args.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        model_kwargs["device_map"] = device_map
    elif device_map:
        model_kwargs["device_map"] = device_map

    return mode


def should_move_model_to_device(mode: str, device_map: str) -> bool:
    return mode in ("fp16", "bf16") and not device_map


def quantization_metadata(args, mode: Optional[str] = None) -> Dict:
    mode = mode or resolve_quantization(args)
    dtype = model_torch_dtype(mode)
    return {
        "quantization": mode,
        "torch_dtype": str(dtype).replace("torch.", ""),
        "load_in_8bit": mode == "int8",
        "load_in_4bit": mode == "int4",
        "bnb_4bit_compute_dtype": (
            getattr(args, "bnb_4bit_compute_dtype", "") if mode == "int4" else ""
        ),
        "bnb_4bit_quant_type": (
            getattr(args, "bnb_4bit_quant_type", "") if mode == "int4" else ""
        ),
        "bnb_4bit_use_double_quant": (
            getattr(args, "bnb_4bit_use_double_quant", False) if mode == "int4" else False
        ),
        "device_map": effective_device_map(mode, getattr(args, "device_map", "")),
    }
