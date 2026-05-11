"""Hardware-profile proxy reward utilities for LTD policy training/eval.

The coefficients are expressed in rough "target-token equivalent" units. They
are deliberately structural rather than wall-clock measurements so a policy can
learn how the preferred draft tree changes with a hardware profile without
binding the reward to one GPU's latency curve.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional


COEFF_KEYS = ("draft", "verify", "depth", "kv", "launch", "fixed")


PROFILE_LIBRARY: Dict[str, Dict[str, float]] = {
    "balanced": {
        "draft": 0.026,
        "verify": 0.070,
        "depth": 0.050,
        "kv": 0.020,
        "launch": 0.090,
        "fixed": 0.060,
    },
    "h100": {
        "draft": 0.018,
        "verify": 0.055,
        "depth": 0.035,
        "kv": 0.012,
        "launch": 0.060,
        "fixed": 0.040,
    },
    "a5090": {
        "draft": 0.022,
        "verify": 0.065,
        "depth": 0.040,
        "kv": 0.016,
        "launch": 0.055,
        "fixed": 0.045,
    },
    "rtxa6000": {
        "draft": 0.035,
        "verify": 0.090,
        "depth": 0.065,
        "kv": 0.030,
        "launch": 0.140,
        "fixed": 0.080,
    },
    "launch_bound": {
        "draft": 0.026,
        "verify": 0.070,
        "depth": 0.050,
        "kv": 0.020,
        "launch": 0.180,
        "fixed": 0.100,
    },
    "memory_bound": {
        "draft": 0.032,
        "verify": 0.085,
        "depth": 0.055,
        "kv": 0.040,
        "launch": 0.100,
        "fixed": 0.070,
    },
}


PROFILE_ALIASES = {
    "rtx_a6000": "rtxa6000",
    "a6000": "rtxa6000",
    "rtx6000": "rtxa6000",
    "5090": "a5090",
    "h100_bf16": "h100",
    "a5090_bf16": "a5090",
    "bf16_hwproxy": "balanced",
    "hwproxy_bf16": "balanced",
}


RANDOM_PROFILE_RANGES = {
    "draft": (0.015, 0.045),
    "verify": (0.045, 0.120),
    "depth": (0.025, 0.080),
    "kv": (0.008, 0.040),
    "launch": (0.040, 0.180),
    "fixed": (0.020, 0.100),
}


PROFILE_VECTOR_SCALES = {
    "draft": 0.030,
    "verify": 0.080,
    "depth": 0.055,
    "kv": 0.025,
    "launch": 0.110,
    "fixed": 0.065,
}


@dataclass(frozen=True)
class HardwareCost:
    draft: float
    verify: float
    depth: float
    kv: float
    launch: float
    fixed: float
    name: str = "custom"

    def as_dict(self) -> Dict[str, float]:
        return {key: float(getattr(self, key)) for key in COEFF_KEYS}


def _canonical_profile_name(name: str) -> str:
    clean = (name or "balanced").strip().lower().replace("-", "_")
    return PROFILE_ALIASES.get(clean, clean)


def parse_cost_coeffs(raw: str) -> Dict[str, float]:
    """Parse comma-separated KEY=VALUE coefficients."""
    if not raw:
        return {}
    coeffs: Dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid hardware cost coefficient {item!r}; use KEY=VALUE pairs."
            )
        key, value = item.split("=", 1)
        key = key.strip().lower()
        if key not in COEFF_KEYS:
            raise ValueError(
                f"Unknown hardware cost coefficient {key!r}; expected one of {COEFF_KEYS}."
            )
        coeffs[key] = float(value)
    return coeffs


def _log_uniform(low: float, high: float, rng=random) -> float:
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_random_profile(rng=random) -> HardwareCost:
    values = {
        key: _log_uniform(low, high, rng)
        for key, (low, high) in RANDOM_PROFILE_RANGES.items()
    }
    return HardwareCost(**values, name="randomized")


def resolve_hardware_profile(
    profile_name: str = "balanced",
    cost_coeffs: str = "",
    randomize: bool = False,
    rng=random,
) -> HardwareCost:
    if randomize or _canonical_profile_name(profile_name) in {"random", "randomized"}:
        return sample_random_profile(rng)

    name = _canonical_profile_name(profile_name)
    if name not in PROFILE_LIBRARY:
        known = ", ".join(sorted(PROFILE_LIBRARY))
        raise ValueError(f"Unknown hardware profile {profile_name!r}; known profiles: {known}")

    values: MutableMapping[str, float] = dict(PROFILE_LIBRARY[name])
    values.update(parse_cost_coeffs(cost_coeffs))
    return HardwareCost(**values, name=name if not cost_coeffs else f"{name}+custom")


def hardware_profile_vector(profile: HardwareCost):
    """Return normalized hardware features in COEFF_KEYS order."""
    values = profile.as_dict()
    return [values[key] / PROFILE_VECTOR_SCALES[key] for key in COEFF_KEYS]


def profile_to_log_dict(profile: HardwareCost) -> Dict[str, float]:
    data = profile.as_dict()
    data["profile_name"] = profile.name
    return data


def proxy_cost(
    profile: HardwareCost,
    drafted_tokens: float,
    verify_tokens: float,
    depth_steps: float,
    kv_tokens: float,
    launch_steps: float,
) -> float:
    return max(
        profile.fixed
        + profile.draft * float(drafted_tokens)
        + profile.verify * float(verify_tokens)
        + profile.depth * float(depth_steps)
        + profile.kv * float(kv_tokens)
        + profile.launch * float(launch_steps),
        1e-6,
    )


def proxy_reward(
    accepted_tokens: float,
    profile: HardwareCost,
    drafted_tokens: float,
    verify_tokens: float,
    depth_steps: float,
    kv_tokens: float,
    launch_steps: float,
    reward_scale: float = 1.0,
) -> tuple[float, float]:
    cost = proxy_cost(
        profile,
        drafted_tokens=drafted_tokens,
        verify_tokens=verify_tokens,
        depth_steps=depth_steps,
        kv_tokens=kv_tokens,
        launch_steps=launch_steps,
    )
    return float(reward_scale) * (float(accepted_tokens) / cost), cost


def infer_policy_obs_size(policy, fallback: int) -> int:
    try:
        return int(policy.observation_space.shape[0])
    except Exception:
        return fallback
