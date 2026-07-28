"""The ADR-013 rule 3 quality gate for ``Fidelity.ALTERED`` stages.

> Lossy stages are gated by evals... they may not ship without a quality
> suite that runs in CI and fails the build on regression.

That suite is this package plus ``tests/optimize/test_eval.py``, which
imports it and asserts every case passes -- an ordinary pytest test is
already this project's CI gate (property tests, fail-inject, contract tests
all work the same way), so no separate runner or CLI is needed for the
requirement itself.

**Deliberately model-free.** A lossy stage can only damage correctness by
removing information from what reaches the model, so checking that required
text survives a stage's transformation is a direct test of the actual
failure mode, not a proxy for one -- and it costs nothing, runs in
milliseconds, and never flakes on model sampling the way grading a live
answer with an LLM judge would (the "who evals the evaluator" trap
``docs/quality.md`` already names for the core's quality lane).

**What this cannot validate.** Fact survival in the prompt is necessary but
not sufficient for a good answer -- a real model could still misuse
preserved facts, and ``route_models``'s risk (a genuinely hard step sent to
a weaker model) is about model capability, which no prompt-level check can
see. Where that matters, ``bench/harness.py``'s live A/B path with a
caller-supplied judge is the tool, not this module -- see each case type's
docstring for which half of the problem it covers.
"""

from __future__ import annotations
