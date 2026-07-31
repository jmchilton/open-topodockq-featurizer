"""Open clean-room TopoDockQ interface featurizer.

Emits the raw 2,754-value channel-major interface descriptor consumed by the (open, MIT) TopoDockQ
scorer's ``03_extract_features_from_npy_to_csv.py``. See README for layout, constants, and provenance.
"""

CHANNELS = ("CC", "CN", "CO", "NC", "NN", "NO", "OC", "ON", "OO")
PER_CHANNEL_WIDTH = 306
RAW_WIDTH = 2754

__all__ = ["CHANNELS", "PER_CHANNEL_WIDTH", "RAW_WIDTH"]
