"""Open clean-room TopoDockQ interface featurizer.

Emits the raw 2,754-value channel-major interface descriptor consumed by the (open, MIT) TopoDockQ
scorer's ``03_extract_features_from_npy_to_csv.py``. See README for layout, constants, and provenance.
"""

CHANNELS = ("CC", "CN", "CO", "NC", "NN", "NO", "OC", "ON", "OO")

from .featurize import (  # noqa: E402
    PER_CHANNEL_WIDTH,
    RAW_WIDTH,
    featurize_channel,
    featurize_interface,
)

__all__ = [
    "CHANNELS",
    "PER_CHANNEL_WIDTH",
    "RAW_WIDTH",
    "featurize_interface",
    "featurize_channel",
]
