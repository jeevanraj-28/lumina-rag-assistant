## 2024-05-19 - XSS Vulnerability in JS event handlers
**Vulnerability:** XSS via unescaped string injection in inline JS event handlers (`onclick`).
**Learning:** `escapeHtml` alone is not enough to prevent injection in inline JS. You need to escape Javascript syntax like `\`, `'`, `"`, `\n`, `\r`, `\t` first so that the context of the script does not break.
**Prevention:** Always use `escapeJs` in conjunction with `escapeHtml` when putting user-controlled strings into Javascript event attributes in HTML.
