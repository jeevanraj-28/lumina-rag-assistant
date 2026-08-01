## 2026-08-01 - Missing ARIA Labels on Modal Close Buttons
**Learning:** Found a specific accessibility pattern where multiple modals and interactive UI panels use icon-only close buttons without an `aria-label`. This makes it difficult for screen readers to identify the button's purpose.
**Action:** Always verify that icon-only buttons (especially close or dismiss buttons) contain appropriate `aria-label` attributes for proper accessibility.
