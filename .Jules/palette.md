## 2024-05-24 - [Modal Keyboard Nav & Icon A11y]
**Learning:** Hardcoding `Escape` key close logic for individual modals creates inconsistent navigation, while missing ARIA labels on icon buttons forces screen readers to read literal icon text (e.g. "close"), confusing users.
**Action:** Always map the Escape key to close all active dialogs generically (e.g., using `document.querySelectorAll(".modal-backdrop:not(.hidden)")`) and add `aria-label` / `aria-hidden="true"` to custom icon buttons to ensure smooth and predictable keyboard and screen reader accessibility.
