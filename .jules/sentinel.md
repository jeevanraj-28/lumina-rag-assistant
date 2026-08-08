## 2024-05-18 - [Fix XSS in inline JS event handlers]
**Vulnerability:** Cross-Site Scripting (XSS) due to HTML-escaping user input that was being embedded in inline JS event handlers (e.g. `onclick="func('${...}')"`).
**Learning:** HTML attributes are decoded *before* JavaScript parsing in the browser. Therefore, an `escapeHtml()` function does not protect against XSS if the output is executed inside JS within an HTML attribute, because the browser parses the HTML escape sequences first.
**Prevention:** Dynamic values in inline handlers must be BOTH JavaScript-escaped AND HTML-escaped. Implement an `escapeJs()` function to sanitize JS control characters (quotes, backslashes, newlines) before applying HTML entity encoding.
