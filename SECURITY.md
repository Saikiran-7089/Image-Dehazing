# Security Policy

## Reporting Vulnerabilities

We take the security of the **AI-Based Single Image Dehazing System** seriously. If you discover a security vulnerability or path traversal issue, please do NOT report it on public issue trackers.

Instead, please send a security report via email to `security@dehazeai.org`.

## Security Features

- **Path Traversal Protection**: Output filenames and zip exports are sanitized using `Path.name` filtering and `is_relative_to()` directory boundary checks.
- **Buffer Limits**: Input file sizes are validated to prevent memory exhaustion attacks.
- **Autograd Memory Cleanup**: Tensor forward passes run under `@torch.inference_mode()`.
