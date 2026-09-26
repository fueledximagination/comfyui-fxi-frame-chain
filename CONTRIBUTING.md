# Contributing

Thanks for helping. This node pack is meant to stay small and readable.

## Setup

```bash
pip install torch pytest ruff   # CPU torch is enough
pytest
ruff check . && ruff format --check .
```

## Good contributions

- Bug fixes in last-frame extraction, chaining, stitching, or the speed ramp.
- Example workflows you have actually run, built from core nodes and public models, with a note on each model's license.
- Docs that make frame chaining easier to understand.

## Ground rules

- Every change ships with tests that run on CPU with tiny tensors: no GPU, no ComfyUI install, no network.
- Keep tensor logic in `fxi_frame_chain/core.py` (no ComfyUI imports) and the node layer in `fxi_frame_chain/nodes.py`.
- No new runtime dependencies beyond what ComfyUI ships. Propose one in an issue first if you think it's needed.
- Node class names and input names are a public contract: renaming them breaks saved workflows.
- Never commit secrets, private hostnames or IP addresses, personal file paths, or model weights.

## Pull requests

1. Fork, branch, and keep the change focused.
2. Run `pytest` and `ruff`.
3. Describe what changed and how you tested it.
