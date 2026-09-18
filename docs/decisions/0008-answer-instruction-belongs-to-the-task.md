# 0008: The answer instruction belongs to the task, and follows the item

**Status:** accepted 2026-09-18 (owner request, refining decision 0007)

## Decisions

1. **`answer_instruction` is a field of the task, not of the prompt format.**
   - `scoring_task(..., answer_instruction =)` in R, `ScoringTask(..., answer_instruction=)` in Python, `task.answer_instruction` on the wire, next to `instructions`.
   - Reason: like `instructions`, it says what is being asked, so researchers can put a reminder of the task itself there ("Reply with the number that best describes how well the statement fits you."). The prompt format keeps only the templates, which say where things go.
   - It may be empty. `{answer_instruction}` remains a placeholder in `system` and `user`; the `unused_answer_instruction` warning now reports a task field.
   - `PromptFormat`/`prompt_format()` reject `answer_instruction` as an unknown field, and a task JSON with `prompt.answer_instruction` is `422 invalid_task`.
2. **The default places it after the item**, in the user message:
   - `system` = `{instructions}\n\n{scale}`
   - `user` = `Text:\n<text>\n{item}\n</text>\n\n{answer_instruction}`
   - It is therefore repeated for every worked example, because examples render through the same `user` template.
   - The previous placement stays available: `prompt_format(system = "{instructions}\n\n{scale}\n\n{answer_instruction}", user = "Text:\n<text>\n{item}\n</text>")`.
3. **Renderer version 3**, so the engine fingerprint changes and prepared tasks and checkpoints from earlier versions must be prepared again. Decision 0007's point 2 (defaults reproduce renderer 1 byte for byte) no longer holds and is superseded: the default prompt now differs deliberately.
4. **The fidelity fixtures pin their prompts.** The two fixtures in `tests/fixtures/fidelity/` now carry an explicit `prompt` block and `answer_instruction` reproducing the prompts their reference tokens and probabilities were computed from. The fixtures are evidence about the engine's numerics, so they must not move when a default changes. A check confirmed that all 478 rendered prompts are byte-identical to the ones recorded, so the reference values still apply and no GPU rerun is needed for this change.

## Evidence (2026-09-18)

- **Python:** 235 tests pass, including the service prompt tests and the client's offline messages against all 44 fixture message cases.
- **R:** `devtools::test()` passes, 279 checks, including messages-fixture parity and both cross-language checkpoint fixtures.
- **Fidelity fixtures:** 450 + 28 items re-rendered; every prompt identical to the recorded one after pinning.
- **Not executed:** the F32 fidelity gate on the GPU and the container acceptance test (the owner's container holds the GPU and port). The engine fingerprint changes with the renderer version, so the running service must be rebuilt and recreated before clients from this version can use it.
