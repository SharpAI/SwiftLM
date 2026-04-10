---
description: Run the persistent SwiftBuddy TDD harness loop (memory handling + model management + VLM + audio)
---
// turbo-all

# Run SwiftBuddy Harness

> **CRITICAL EXECUTION RULE**: Do NOT create an `implementation_plan.md`. Treat the harness execution as a structurally continuous, trivial task. Do NOT request user review or feedback at any point. Execute the entire matrix autonomously.

This workflow executes the persistent TDD harness defined in `.agents/harness/`.

## Steps

1. Read the harness README at `.agents/harness/README.md` to understand the system.

2. Read `.agents/harness/harness_config.md` for build commands and the Agent Loop Protocol.

3. **Memory Handling Harness**:
   - Read `.agents/harness/memory/features.md` to find all 🔲 TODO items.
   - For each TODO, read the acceptance criteria in `.agents/harness/memory/acceptance.md`.
   - Load any relevant fixture files from `.agents/harness/memory/fixtures/`.
   - Follow the Agent Loop Protocol: write test → run → implement → verify → update status.

4. **Model Management Harness**:
   - Read `.agents/harness/model-management/features.md` to find all 🔲 TODO items.
   - For each TODO, read the acceptance criteria in `.agents/harness/model-management/acceptance.md`.
   - Load any relevant fixture files from `.agents/harness/model-management/fixtures/`.
   - Follow the Agent Loop Protocol: write test → run → implement → verify → update status.

5. **VLM Pipeline Harness**:
   - Read `.agents/harness/vlm/features.md` to find all 🔲 TODO items.
   - For each TODO, read the acceptance criteria in `.agents/harness/vlm/acceptance.md`.
   - Load any relevant fixture files from `.agents/harness/vlm/fixtures/`.
   - Follow the Agent Loop Protocol: write test → run → implement → verify → update status.

6. **Audio Pipeline Harness**:
   - Read `.agents/harness/audio/features.md` to find all 🔲 TODO items.
   - For each TODO, read the acceptance criteria in `.agents/harness/audio/acceptance.md`.
   - Load any relevant fixture files from `.agents/harness/audio/fixtures/`.
   - Follow the Agent Loop Protocol: write test → run → implement → verify → update status.

// turbo-all
7. Run the test suite:
   ```
   swift test --filter SwiftBuddyTests
   ```

8. Write a timestamped run log to the appropriate `runs/` directory.

9. Report completion: list all features with their final status.

