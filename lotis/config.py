"""Configuration handling for LoTIS models."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union
import yaml


@dataclass
class ModelConfig:
    """Configuration for the LoTIS model."""

    hidden_dim: int = 512
    num_heads: int = 8
    num_blocks: int = 3
    head_depth: int = 2
    max_seq_len: int = 40
    rope_freq_seq: int = 100
    rope_freq_spat: int = 500
    prediction_heads: List[str] = field(default_factory=lambda: ["center"])
    mini_batch_size: int = 8
    layernorm_type: str = "LayerNorm"


def load_config(config_path: Union[str, Path]) -> ModelConfig:
    """
    Load model configuration from a YAML file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        ModelConfig instance with loaded values.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Handle nested YACS-style config
    model_cfg = raw_config.get("MODEL", raw_config)
    general_cfg = raw_config.get("GENERAL", {})
    training_cfg = raw_config.get("TRAINING", {})

    return ModelConfig(
        hidden_dim=model_cfg.get("HIDDEN_DIM", 512),
        num_heads=model_cfg.get("NUM_HEADS", 8),
        num_blocks=model_cfg.get("NUM_BLOCKS", 3),
        head_depth=model_cfg.get("HEAD_DEPTH", 2),
        max_seq_len=model_cfg.get("MAX_SEQ_LEN", 40),
        rope_freq_seq=model_cfg.get("ROPE_FREQ_SEQ", 100),
        rope_freq_spat=model_cfg.get("ROPE_FREQ_SPAT", 500),
        prediction_heads=model_cfg.get("PREDICTION_HEADS", ["center"]),
        mini_batch_size=model_cfg.get("MINI_BATCH_SIZE", 8),
        layernorm_type=model_cfg.get("LAYERNORM_TYPE", "LayerNorm"),
    )
