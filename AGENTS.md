# Gemini Code Assist Instructions

## 🛑 CRITICAL RULE: Parameter & Configuration Integrity
* **Do not modify configuration constants and parameters** unless I explicitly ask you to.
* If a request is about the **body** of a function or a specific logic flow, keep the signature and any associated config objects exactly as they are.
* If you believe a parameter change is necessary to fulfill a request, **ask for permission** first or provide it as a separate suggestion.
* Always use parameters for hardcoded values

## Coding Style & Standards
* **Python Version:** Target Python 3.12+ features (e.g., advanced type hinting) and follow best practices such as list instead of importing typing.
* **Type Safety:** Always include PEP 484 type hints for function arguments and return types.
* **Docstrings:** Use the Google Python Style Guide format for all docstrings.
* **Refactoring:** Prioritize code readability and maintainability. Avoid "clever" one-liners if they obscure the logic.

## Context Awareness
* Before suggesting a change, scan the existing project structure to ensure naming consistency.

## Explainability
* When making a change also provide the explanation of what the code does

## Comments
* Do not hardcode numbers into comments
