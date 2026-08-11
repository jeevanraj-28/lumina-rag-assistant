## 2026-08-11 - [Dropdown Accessibility]
**Learning:** Adding aria-expanded dynamically to interactive UI elements like dropdowns is crucial to support screen readers, allowing them to know the current state of the dropdown.
**Action:** Always add aria-expanded="false" initially to dropdown buttons and update it to "true" when expanded dynamically using JavaScript.
