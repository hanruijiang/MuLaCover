"""Load YourMT3 tensor weights."""

import numpy as np
import torch


class _CheckpointMetadata:
    """Inert holder for discarded tokenizer metadata."""


def load_state_dict(path):
    """Read the state dictionary, ignoring tokenizer metadata."""
    aliases = [
        (_CheckpointMetadata, name)
        for name in (
            "utils.task_manager.TaskManager",
            "utils.tokenizer.NoteEventTokenizer",
            "utils.note_event_dataclasses.EventRange",
            "utils.event_codec.FastCodec",
            "utils.note_event_dataclasses.Event",
        )
    ]
    core = np._core
    aliases.extend(
        [
            core.multiarray._reconstruct,
            core.multiarray.scalar,
            (core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
            (core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            np.ndarray,
            np.dtype,
            type(np.dtype("int64")),
            type(np.dtype("float64")),
            type(np.dtype("int32")),
            type(np.dtype("float32")),
        ]
    )
    with torch.serialization.safe_globals(aliases):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("YourMT3 checkpoint must contain a state dictionary")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("YourMT3 checkpoint does not contain a state dictionary")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state.items()
    ):
        raise ValueError("YourMT3 state dictionary must contain named tensors only")
    return {
        key: value for key, value in state.items() if not key.startswith("pitchshift.")
    }
