## 2025-02-18 - [Defense in Depth Path Traversal]
**Vulnerability:** The `/files/{path:path}` endpoint validated paths using `Path.is_relative_to` after resolution. While secure, it lacked early input validation, exposing the system to potential security scanner alerts and relying solely on post-resolution checks.
**Learning:** Adding explicit early string validation (`../`, `/..`, `/`) provides a defense-in-depth layer, failing securely with a 400 Bad Request before hitting the filesystem resolver.
**Prevention:** Always validate and sanitize user input *before* passing it to file system resolution functions, even if the resolver has built-in protections.
