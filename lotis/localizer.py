"""
Main TrajectoryLocalizer class - the primary user-facing API.
"""

from pathlib import Path
from typing import Union, List, Optional, overload
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from .result import LocalizationResult
from .preprocessing import preprocess_image, load_images_from_path
from .feature_extraction import setup_feature_extractor, extract_features


QueryInput = Union[str, Path, Image.Image, np.ndarray]


@dataclass
class TrajectoryEncoding:
    """
    Encoded trajectory representation that can be stored and reused.

    Example:
        >>> encoding = localizer.encode_trajectory("video.mp4")
        >>> torch.save(encoding.to_dict(), "encoding.pt")
        >>> # Later...
        >>> encoding = TrajectoryEncoding.from_dict(torch.load("encoding.pt"))
        >>> result = localizer.localize("query.jpg", encoding)
    """
    features: torch.Tensor  # Encoded trajectory features [1, S, P, C]
    seq_len: int  # Number of frames in the trajectory
    _seq_lens_tensor: torch.Tensor  # Internal: sequence lengths tensor for model

    def to_dict(self) -> dict:
        """Convert to a dictionary for saving with torch.save()."""
        return {
            "features": self.features.cpu(),
            "seq_len": self.seq_len,
            "_seq_lens_tensor": self._seq_lens_tensor.cpu(),
        }

    @classmethod
    def from_dict(cls, data: dict, device: str = "cuda") -> "TrajectoryEncoding":
        """Load from a dictionary (e.g., from torch.load())."""
        return cls(
            features=data["features"].to(device),
            seq_len=data["seq_len"],
            _seq_lens_tensor=data["_seq_lens_tensor"].to(device),
        )

    def to(self, device: str) -> "TrajectoryEncoding":
        """Move encoding to a different device."""
        return TrajectoryEncoding(
            features=self.features.to(device),
            seq_len=self.seq_len,
            _seq_lens_tensor=self._seq_lens_tensor.to(device),
        )


@dataclass
class BatchedTrajectoryEncoding:
    """
    Multiple trajectory encodings pre-concatenated for efficient batch inference.

    Use this when you have a fixed set of trajectories and want to avoid
    re-concatenating tensors on every localize() call.

    Example:
        >>> encodings = [localizer.encode_trajectory(f"traj_{i}.mp4") for i in range(100)]
        >>> batched = BatchedTrajectoryEncoding.from_encodings(encodings)
        >>> # Now use batched for all queries - no re-concatenation needed
        >>> results = localizer.localize(queries, batched, query_to_trajectory=mapping)
    """
    features: torch.Tensor  # Concatenated features [N, S, P, C]
    seq_lens: torch.Tensor  # Sequence lengths [N]
    frame_counts: List[int]  # Number of frames per trajectory
    num_trajectories: int

    @classmethod
    def from_encodings(
        cls,
        encodings: List[TrajectoryEncoding],
        device: Optional[str] = None,
    ) -> "BatchedTrajectoryEncoding":
        """
        Create a batched encoding from a list of individual encodings.

        Args:
            encodings: List of TrajectoryEncoding objects.
            device: Device to place tensors on. If None, uses device of first encoding.
        """
        if not encodings:
            raise ValueError("Must provide at least one encoding")

        device = device or encodings[0].features.device

        features = torch.cat([enc.features.to(device) for enc in encodings], dim=0)
        seq_lens = torch.cat([enc._seq_lens_tensor.to(device) for enc in encodings], dim=0)
        frame_counts = [enc.seq_len for enc in encodings]

        return cls(
            features=features,
            seq_lens=seq_lens,
            frame_counts=frame_counts,
            num_trajectories=len(encodings),
        )

    def to_dict(self) -> dict:
        """Convert to a dictionary for saving with torch.save()."""
        return {
            "features": self.features.cpu(),
            "seq_lens": self.seq_lens.cpu(),
            "frame_counts": self.frame_counts,
            "num_trajectories": self.num_trajectories,
        }

    @classmethod
    def from_dict(cls, data: dict, device: str = "cuda") -> "BatchedTrajectoryEncoding":
        """Load from a dictionary (e.g., from torch.load())."""
        return cls(
            features=data["features"].to(device),
            seq_lens=data["seq_lens"].to(device),
            frame_counts=data["frame_counts"],
            num_trajectories=data["num_trajectories"],
        )

    def to(self, device: str) -> "BatchedTrajectoryEncoding":
        """Move encoding to a different device."""
        return BatchedTrajectoryEncoding(
            features=self.features.to(device),
            seq_lens=self.seq_lens.to(device),
            frame_counts=self.frame_counts,
            num_trajectories=self.num_trajectories,
        )

    def __len__(self) -> int:
        return self.num_trajectories

    def __getitem__(self, idx: int) -> TrajectoryEncoding:
        """Get a single encoding by index."""
        if idx < 0 or idx >= self.num_trajectories:
            raise IndexError(f"Index {idx} out of range for {self.num_trajectories} trajectories")
        return TrajectoryEncoding(
            features=self.features[idx : idx + 1],
            seq_len=self.frame_counts[idx],
            _seq_lens_tensor=self.seq_lens[idx : idx + 1],
        )


class TrajectoryLocalizer:
    """
    Stateless interface for trajectory localization.

    Example:
        >>> localizer = TrajectoryLocalizer.from_checkpoint("model.pth", "config.yaml")
        >>> encoding = localizer.encode_trajectory("trajectory.mp4")
        >>> result = localizer.localize("query.jpg", encoding)
        >>> print(f"Closest frame: {result.closest_frame()}")

    Batching examples:
        >>> # Single query against multiple trajectories
        >>> results = localizer.localize("query.jpg", [enc1, enc2, enc3])

        >>> # Multiple queries against single trajectory
        >>> results = localizer.localize(["q1.jpg", "q2.jpg"], encoding)

        >>> # Multiple queries against multiple trajectories (cartesian product)
        >>> results = localizer.localize(["q1.jpg", "q2.jpg"], [enc1, enc2])
        >>> # Returns 4 results: [(q1,enc1), (q1,enc2), (q2,enc1), (q2,enc2)]

        >>> # Explicit mapping: specific query-trajectory pairs
        >>> results = localizer.localize(
        ...     ["q1.jpg", "q2.jpg", "q3.jpg"],
        ...     [enc1, enc2],
        ...     query_to_trajectory=[0, 0, 1]  # q1,q2->enc1, q3->enc2
        ... )
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_extractor: torch.nn.Module,
        device: str = "cuda",
        max_seq_len: int = 40,
        use_amp: bool = True,
    ):
        """
        Initialize the localizer with a model and feature extractor.

        Use TrajectoryLocalizer.from_checkpoint() for easier initialization.
        """
        self.model = model.to(device).eval()
        self.feature_extractor = feature_extractor.to(device).eval()
        self.device = device
        self.max_seq_len = max_seq_len
        self.use_amp = use_amp and device == "cuda"

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config_path: Union[str, Path],
        device: str = "cuda",
        feature_extractor_type: str = "dinov3",
        dinov3_weights: Optional[str] = None,
        dinov3_repo: str = "./dinov3",
    ) -> "TrajectoryLocalizer":
        """
        Create a TrajectoryLocalizer from a checkpoint and config file.

        Args:
            checkpoint_path: Path to the model checkpoint (.pth file).
            config_path: Path to the model config (.yaml file).
            device: Device to run inference on.
            feature_extractor_type: Type of feature extractor ('dinov3' or 'dinov2').
            dinov3_weights: Path to DINOv3 weights (.pth). Falls back to the
                DINOV3_WEIGHTS environment variable.
            dinov3_repo: Path to the cloned facebookresearch/dinov3 repo
                (default: "./dinov3"). Falls back to the DINOV3_REPO env var.

        Returns:
            Initialized TrajectoryLocalizer ready for use.
        """
        from .model import TrajectoryLocalizationModel
        from .config import load_config

        cfg = load_config(config_path)

        model = TrajectoryLocalizationModel(
            feature_dim=768,
            input_patches=(14, 14),
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_blocks=cfg.num_blocks,
            head_depth=cfg.head_depth,
            dropout=0.0,
            attention_dropout=0.0,
            droppath=0.0,
            max_seq_len=cfg.max_seq_len,
            output_size=(32, 32),
            full_global_attention=True,
            rope_freq_seq=cfg.rope_freq_seq,
            rope_freq_spat=cfg.rope_freq_spat,
            heads=cfg.prediction_heads,
            use_nested_tensor=False,
            compile=False,
            mini_batch_size=cfg.mini_batch_size,
            layernorm_type=cfg.layernorm_type,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        state_dict = {}
        for key, value in checkpoint.items():
            new_key = key.replace("._orig_mod", "")
            state_dict[new_key] = value

        model.load_state_dict(state_dict, strict=False)
        model = model.eval()
        feature_extractor = setup_feature_extractor(
            feature_extractor_type,
            dinov3_weights=dinov3_weights,
            dinov3_repo=dinov3_repo,
        )

        return cls(
            model=model,
            feature_extractor=feature_extractor,
            device=device,
            max_seq_len=cfg.max_seq_len,
        )

    def encode_trajectory(
        self,
        trajectory: Union[str, Path, List[Image.Image], List[np.ndarray]],
        max_frames: Optional[int] = None,
    ) -> TrajectoryEncoding:
        """
        Encode a trajectory for later localization.

        Args:
            trajectory: Path to video, path to image directory, list of PIL Images,
                       or list of numpy arrays (RGB, HWC format).
            max_frames: Maximum frames to use (subsamples if exceeded).

        Returns:
            TrajectoryEncoding that can be reused, saved, and loaded.
        """
        max_frames = max_frames or self.max_seq_len

        if isinstance(trajectory, (str, Path)):
            images = load_images_from_path(trajectory, max_images=max_frames)
        else:
            images = trajectory
            if len(images) > max_frames:
                indices = np.linspace(0, len(images) - 1, max_frames, dtype=int)
                images = [images[i] for i in indices]

        if not images:
            raise ValueError("No images found in trajectory")

        frame_count = len(images)
        processed = torch.stack([preprocess_image(img) for img in images]).to(self.device)

        with torch.no_grad():
            features = extract_features(processed, self.feature_extractor, self.device)
            features = features.unsqueeze(0)  # [1, S, H, W, C]

            seq_lens = torch.tensor([frame_count], dtype=torch.int64, device=self.device)

            with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.use_amp):
                encoded = self.model.encode_trajectory(features, seq_lens)

        return TrajectoryEncoding(
            features=encoded,
            seq_len=frame_count,
            _seq_lens_tensor=seq_lens,
        )

    def localize(
        self,
        query: Union[QueryInput, List[QueryInput]],
        encoding: Union[TrajectoryEncoding, List[TrajectoryEncoding], BatchedTrajectoryEncoding],
        query_to_trajectory: Optional[List[int]] = None,
    ) -> Union[LocalizationResult, List[LocalizationResult]]:
        """
        Localize query image(s) within trajectory encoding(s).

        Behavior depends on inputs:
        - Single query + single encoding: returns single LocalizationResult
        - Single query + multiple encodings: query against all encodings
        - Multiple queries + single encoding: all queries against that encoding
        - Multiple queries + multiple encodings:
          - With query_to_trajectory: use explicit mapping
          - Without: cartesian product (each query against each encoding)

        Args:
            query: Single query or list of queries.
                Accepts: path (str/Path), PIL Image, or numpy array (RGB uint8, HWC).
                For numpy arrays from OpenCV, convert BGR to RGB first:
                `cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)`
            encoding: Single TrajectoryEncoding, list of encodings, or BatchedTrajectoryEncoding.
                Use BatchedTrajectoryEncoding for efficiency when reusing the same set of
                trajectories across multiple localize() calls.
            query_to_trajectory: Optional explicit mapping from query index to encoding index.

        Returns:
            Single LocalizationResult if single query + single encoding,
            otherwise List[LocalizationResult].

        Examples:
            >>> # Single query, single trajectory
            >>> result = localizer.localize("query.jpg", encoding)

            >>> # Single query against 3 trajectories
            >>> results = localizer.localize("query.jpg", [enc1, enc2, enc3])

            >>> # 3 queries against same trajectory
            >>> results = localizer.localize(["q1.jpg", "q2.jpg", "q3.jpg"], encoding)

            >>> # Cartesian product: 2 queries x 2 trajectories = 4 results
            >>> results = localizer.localize(["q1.jpg", "q2.jpg"], [enc1, enc2])

            >>> # Explicit mapping
            >>> results = localizer.localize(
            ...     ["q1.jpg", "q2.jpg", "q3.jpg"],
            ...     [enc1, enc2],
            ...     query_to_trajectory=[0, 0, 1]
            ... )

            >>> # Efficient batch with pre-concatenated encodings
            >>> batched = BatchedTrajectoryEncoding.from_encodings([enc1, enc2, enc3])
            >>> results = localizer.localize(queries, batched, query_to_trajectory=mapping)
        """
        # Handle BatchedTrajectoryEncoding
        if isinstance(encoding, BatchedTrajectoryEncoding):
            return self._localize_with_batched(query, encoding, query_to_trajectory)

        # Normalize inputs to lists
        single_query = not isinstance(query, list)
        single_encoding = isinstance(encoding, TrajectoryEncoding)

        queries = [query] if single_query else query
        encodings = [encoding] if single_encoding else encoding

        if not queries:
            raise ValueError("Must provide at least one query")
        if not encodings:
            raise ValueError("Must provide at least one encoding")

        # Determine query-to-trajectory mapping
        if query_to_trajectory is not None:
            # Explicit mapping provided
            if len(query_to_trajectory) != len(queries):
                raise ValueError(
                    f"query_to_trajectory length ({len(query_to_trajectory)}) "
                    f"must match number of queries ({len(queries)})"
                )
            mapping = query_to_trajectory
        elif len(encodings) == 1:
            # All queries go to the single encoding
            mapping = [0] * len(queries)
        elif len(queries) == 1:
            # Single query goes to all encodings - expand to cartesian
            mapping = list(range(len(encodings)))
            queries = queries * len(encodings)
        else:
            # Cartesian product: each query against each encoding
            expanded_queries = []
            mapping = []
            for q in queries:
                for enc_idx in range(len(encodings)):
                    expanded_queries.append(q)
                    mapping.append(enc_idx)
            queries = expanded_queries

        results = self._localize_with_mapping(queries, encodings, mapping)

        # Return single result if single query + single encoding
        if single_query and single_encoding:
            return results[0]
        return results

    def _localize_with_batched(
        self,
        query: Union[QueryInput, List[QueryInput]],
        batched: BatchedTrajectoryEncoding,
        query_to_trajectory: Optional[List[int]],
    ) -> Union[LocalizationResult, List[LocalizationResult]]:
        """Handle localization with pre-batched encodings."""
        single_query = not isinstance(query, list)
        queries = [query] if single_query else query

        if not queries:
            raise ValueError("Must provide at least one query")

        num_encodings = batched.num_trajectories

        # Determine mapping
        if query_to_trajectory is not None:
            if len(query_to_trajectory) != len(queries):
                raise ValueError(
                    f"query_to_trajectory length ({len(query_to_trajectory)}) "
                    f"must match number of queries ({len(queries)})"
                )
            mapping = query_to_trajectory
        elif num_encodings == 1:
            mapping = [0] * len(queries)
        elif len(queries) == 1:
            mapping = list(range(num_encodings))
            queries = queries * num_encodings
        else:
            # Cartesian product
            expanded_queries = []
            mapping = []
            for q in queries:
                for enc_idx in range(num_encodings):
                    expanded_queries.append(q)
                    mapping.append(enc_idx)
            queries = expanded_queries

        # Count queries per trajectory
        traj_query_counts = [0] * num_encodings
        for traj_idx in mapping:
            if traj_idx < 0 or traj_idx >= num_encodings:
                raise ValueError(f"Invalid trajectory index {traj_idx}")
            traj_query_counts[traj_idx] += 1

        # Sort queries by trajectory index
        sorted_indices = sorted(range(len(queries)), key=lambda i: mapping[i])
        sorted_queries = [queries[i] for i in sorted_indices]

        # Preprocess queries
        processed_queries = []
        for q in sorted_queries:
            if isinstance(q, (str, Path)):
                img = Image.open(q).convert("RGB")
            elif isinstance(q, np.ndarray):
                img = Image.fromarray(q)
            else:
                img = q
            processed_queries.append(preprocess_image(img))

        query_batch = torch.stack(processed_queries).to(self.device)

        # Use pre-batched features directly
        trajectory_features = batched.features.to(self.device)
        seq_lens = batched.seq_lens.to(self.device)

        traj_query_counts_tensor = torch.tensor(
            traj_query_counts, dtype=torch.int32, device=self.device
        )

        with torch.no_grad():
            query_features = extract_features(query_batch, self.feature_extractor, self.device)

            with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.use_amp):
                prediction = self.model(
                    None,
                    None,
                    trajectory_features,
                    seq_lens,
                    query_features,
                    skip_traj_encoding=True,
                    traj_query_counts=traj_query_counts_tensor,
                )

        # Extract results
        sorted_results = []
        for i in range(len(sorted_queries)):
            original_idx = sorted_indices[i]
            traj_idx = mapping[original_idx]
            frame_count = batched.frame_counts[traj_idx]

            coords = prediction["center"]["coords"][i].float().cpu().numpy()
            visibility_logits = prediction["visibility"]["logits"][i].float().cpu().numpy()
            visibility = 1 / (1 + np.exp(-visibility_logits.flatten()))

            distances = None
            if "distances" in prediction and prediction["distances"] is not None:
                distances = prediction["distances"]["values"][i].float().cpu().numpy().flatten()

            sorted_results.append(
                LocalizationResult(
                    coords=coords,
                    visibility=visibility,
                    distances=distances,
                    num_frames=frame_count,
                )
            )

        # Reorder back to original order
        results = [None] * len(queries)
        for sorted_idx, original_idx in enumerate(sorted_indices):
            results[original_idx] = sorted_results[sorted_idx]

        if single_query and num_encodings == 1:
            return results[0]
        return results

    def _localize_with_mapping(
        self,
        queries: List[QueryInput],
        encodings: List[TrajectoryEncoding],
        query_to_trajectory: List[int],
    ) -> List[LocalizationResult]:
        """Internal: localize with explicit query-to-trajectory mapping."""
        # Count queries per trajectory
        num_trajectories = len(encodings)
        traj_query_counts = [0] * num_trajectories
        for traj_idx in query_to_trajectory:
            if traj_idx < 0 or traj_idx >= num_trajectories:
                raise ValueError(f"Invalid trajectory index {traj_idx}")
            traj_query_counts[traj_idx] += 1

        # Sort queries by trajectory index for batching
        sorted_indices = sorted(range(len(queries)), key=lambda i: query_to_trajectory[i])
        sorted_queries = [queries[i] for i in sorted_indices]

        # Load and preprocess queries
        processed_queries = []
        for q in sorted_queries:
            if isinstance(q, (str, Path)):
                img = Image.open(q).convert("RGB")
            elif isinstance(q, np.ndarray):
                img = Image.fromarray(q)
            else:
                img = q
            processed_queries.append(preprocess_image(img))

        query_batch = torch.stack(processed_queries).to(self.device)

        # Stack trajectory encodings
        trajectory_features = torch.cat(
            [enc.features.to(self.device) for enc in encodings], dim=0
        )
        seq_lens = torch.cat(
            [enc._seq_lens_tensor.to(self.device) for enc in encodings], dim=0
        )

        traj_query_counts_tensor = torch.tensor(
            traj_query_counts, dtype=torch.int32, device=self.device
        )

        with torch.no_grad():
            query_features = extract_features(query_batch, self.feature_extractor, self.device)

            with torch.autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.use_amp):
                prediction = self.model(
                    None,
                    None,
                    trajectory_features,
                    seq_lens,
                    query_features,
                    skip_traj_encoding=True,
                    traj_query_counts=traj_query_counts_tensor,
                )

        # Extract results in sorted order
        sorted_results = []
        for i in range(len(sorted_queries)):
            original_idx = sorted_indices[i]
            traj_idx = query_to_trajectory[original_idx]
            frame_count = encodings[traj_idx].seq_len

            coords = prediction["center"]["coords"][i].float().cpu().numpy()
            visibility_logits = prediction["visibility"]["logits"][i].float().cpu().numpy()
            visibility = 1 / (1 + np.exp(-visibility_logits.flatten()))

            distances = None
            if "distances" in prediction and prediction["distances"] is not None:
                distances = prediction["distances"]["values"][i].float().cpu().numpy().flatten()

            sorted_results.append(
                LocalizationResult(
                    coords=coords,
                    visibility=visibility,
                    distances=distances,
                    num_frames=frame_count,
                )
            )

        # Reorder back to original query order
        results = [None] * len(queries)
        for sorted_idx, original_idx in enumerate(sorted_indices):
            results[original_idx] = sorted_results[sorted_idx]

        return results
