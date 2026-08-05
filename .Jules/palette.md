## 2026-08-05 - [Global Escape Key Handler for Modals]
**Learning:** Hardcoding escape key handlers to close specific modals does not scale and breaks keyboard accessibility for users relying on Esc to dismiss dialogs consistently.
**Action:** Use a global window event listener that queries and hides all active `.modal-backdrop:not(.hidden)` elements to provide a robust, unified escape behavior.
