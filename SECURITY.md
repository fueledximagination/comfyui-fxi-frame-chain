# Security Policy

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Report them privately, either way works:

- GitHub private vulnerability reporting: "Report a vulnerability" under this repository's **Security** tab, or
- email **contact@fxi.studio** with "SECURITY: comfyui-fxi-frame-chain" in the subject.

You should get a response within a few days.

## Scope and hardening notes

- The nodes only transform image tensors in memory. They make no network requests, read or write no files, and execute no user-supplied code.
- ComfyUI custom nodes run with the full permissions of your ComfyUI process. Install custom nodes, this one included, only from sources you trust, and review the code before you do.
- Do not expose a ComfyUI instance to the internet without authentication in front of it.

## Supported versions

Only the latest release receives fixes.
