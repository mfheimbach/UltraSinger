"""UltraSinger pipeline: typed artifacts, registered stages, presets (see core.py)."""

import os

# No library telemetry: pyannote (pulled in by whisperx) and the Hugging Face hub phone home by
# default. Set the variables yourself to opt in.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
