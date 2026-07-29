# Plug-and-play cost reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a one-line Anthropic adapter so the package's largest lossless saving becomes reachable and measurable, and add an invariant checker that makes silent quality damage from a default-on stage fail loudly.

**Architecture:** A new `adapters/anthropic.py` mirrors the existing `adapters/openai_agents.py`, wrapping `client.messages.create` on both the sync and async Anthropic clients and routing through the already-tested translation helpers in `wire.py`. A new pure module `invariants.py` holds two rule sets — absolute rules true of any single request, and preservation rules true of any rewrite — with pytest and the real-agent probe as its two callers.

**Tech Stack:** Python 3.10+, `anthropic` SDK, `httpx.MockTransport` for tests, pytest, ruff, mypy --strict, import-linter.

## Global Constraints

Every task's requirements implicitly include this section.

- **No new stages and no new config options.** The stage count is frozen. Adding a row to the `PRICING` dict is data, not a config option, and is permitted.
- **A `Violation` never carries prompt content.** Only `(rule, message_index, role)`. Violations reach terminal scrollback and possibly CI logs; §10's rule applies exactly as it does to the fail-open guard, which logs an exception's type and never its message.
- **`invariants.py` is never called from library code on the request path.** It is a test and probe tool. SC-5's overhead budget is not spent on self-checking.
- **Credentials are never handled.** The adapter takes a client the caller already built. `.env` holds `ANTHROPIC_KEY`; it is mapped to the SDK's own `ANTHROPIC_API_KEY` **in-process only** and never placed on a command line, where a process listing exposes it.
- **Nothing in `pytest` may spend money or touch a network.** Adapter tests use `httpx.MockTransport`.
- **ADR-012:** submodules are internal. Do not add anything to `src/optio_optimize/__init__.py`'s `__all__` in this plan.
- **Fail-open (ADR-013 rule 1):** an adapter translation failure falls back to the unmodified real client.
- **The gate, run from the repo root before every commit:**
  ```bash
  .venv/Scripts/python.exe -m ruff check .
  .venv/Scripts/python.exe -m ruff format --check .
  .venv/Scripts/python.exe -m mypy
  .venv/Scripts/lint-imports.exe
  .venv/Scripts/python.exe -m pytest -q
  ```
  Baseline before this plan starts: **1310 passed, 9 skipped**, ruff clean, mypy clean on 165 files, 4 contracts kept.

---

### Task 1: Move the Anthropic response translator into `wire.py`

`batch_backends._response_from_anthropic_message` is exactly what the new adapter needs. Two copies of it is the specific failure that made a whole benchmark run measure nothing when `tools` went unsent from one of two translation sites. Move it before either caller can diverge.

**Files:**
- Modify: `src/optio_optimize/wire.py` (append)
- Modify: `src/optio_optimize/batch_backends.py:~300-325` (delete the function, import it instead)
- Test: `tests/optimize/test_wire.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `wire.response_from_anthropic_message(message: Any) -> LLMResponse`. Task 4 and Task 5 call it.

- [ ] **Step 1: Write the failing test**

Append to `tests/optimize/test_wire.py`:

```python
class _Usage:
    def __init__(self, input_tokens=300, output_tokens=40, cache_read_input_tokens=100):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = cache_read_input_tokens


class _Block:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _Msg:
    def __init__(self, blocks, usage, model="claude-haiku-4-5", stop_reason="end_turn"):
        self.content = blocks
        self.usage = usage
        self.model = model
        self.stop_reason = stop_reason


def test_anthropic_response_adds_cache_reads_into_input_tokens():
    # Anthropic reports input_tokens EXCLUDING cache reads. Everywhere else in
    # this package input_tokens means "total prompt tokens, some discounted",
    # so getting this wrong in one place makes batched, synchronous and
    # adapter totals silently incomparable.
    from optio_optimize.wire import response_from_anthropic_message

    response = response_from_anthropic_message(_Msg([_Block("hi")], _Usage()))

    assert response.input_tokens == 400
    assert response.cached_input_tokens == 100
    assert response.billable_input_tokens == 300
    assert response.content == "hi"
    assert response.finish_reason == "end_turn"


def test_anthropic_response_joins_only_text_blocks():
    from optio_optimize.wire import response_from_anthropic_message

    class _ToolBlock:
        type = "tool_use"
        text = "SHOULD NOT APPEAR"

    message = _Msg([_Block("part one "), _ToolBlock(), _Block("part two")], _Usage())
    assert response_from_anthropic_message(message).content == "part one part two"


def test_anthropic_response_survives_missing_usage():
    from optio_optimize.wire import response_from_anthropic_message

    class _NoUsage:
        content = [_Block("hi")]
        usage = None
        model = "claude-haiku-4-5"
        stop_reason = "end_turn"

    response = response_from_anthropic_message(_NoUsage())
    assert response.input_tokens == 0
    assert response.output_tokens == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/optimize/test_wire.py -k anthropic_response -v`

Expected: FAIL with `ImportError: cannot import name 'response_from_anthropic_message'`

- [ ] **Step 3: Add the function to `wire.py`**

Append to `src/optio_optimize/wire.py`:

```python
def response_from_anthropic_message(message: Any) -> LLMResponse:
    """Normalize an Anthropic message into this package's response model.

    ``input_tokens`` is reported by Anthropic *excluding* cache reads, so the
    cached count is added back to make the field mean what it means everywhere
    else here: total prompt tokens, of which some were discounted. Getting that
    wrong in one place would make batched, synchronous and adapter totals
    silently incomparable.

    Attribute access rather than isinstance narrowing, because the two callers
    receive structurally identical objects from different SDK code paths --
    a batch result entry's ``.message`` and a live ``messages.create`` return.

    Args:
        message: An ``anthropic.types.Message`` or a batch result's message.

    Returns:
        The normalized response.
    """
    usage = getattr(message, "usage", None)
    cached = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
    text = "".join(
        str(getattr(block, "text", ""))
        for block in getattr(message, "content", [])
        if getattr(block, "type", "") == "text"
    )
    return LLMResponse(
        content=text,
        input_tokens=(int(getattr(usage, "input_tokens", 0)) + cached) if usage else 0,
        output_tokens=int(getattr(usage, "output_tokens", 0)) if usage else 0,
        cached_input_tokens=cached,
        model=str(getattr(message, "model", "")),
        finish_reason=getattr(message, "stop_reason", None),
    )
```

`wire.py` currently imports `LLMRequest` only under `TYPE_CHECKING`. `LLMResponse` is *constructed* here, so it needs a real runtime import. Change the top of `wire.py`:

```python
from optio_optimize.types import LLMResponse

if TYPE_CHECKING:
    from optio_optimize.types import LLMRequest
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/optimize/test_wire.py -k anthropic_response -v`

Expected: 3 passed

- [ ] **Step 5: Delete the duplicate from `batch_backends.py` and import instead**

In `src/optio_optimize/batch_backends.py`, delete the whole `_response_from_anthropic_message` function (the last function in the file) and change the import line:

```python
from optio_optimize.wire import anthropic_body, openai_body, response_from_anthropic_message
```

Then update its one call site in `AnthropicBatchBackend.fetch`:

```python
                responses[custom_id] = response_from_anthropic_message(message)
```

- [ ] **Step 6: Run the batch backend tests to prove nothing broke**

Run: `.venv/Scripts/python.exe -m pytest tests/optimize/test_batch_backends.py -q`

Expected: all pass. `test_anthropic_results_add_cache_reads_back_into_input_tokens` is the one that matters — it exercises the moved function through its original caller.

- [ ] **Step 7: Run the gate and commit**

```bash
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check . && .venv/Scripts/python.exe -m mypy && .venv/Scripts/python.exe -m pytest -q
git add src/optio_optimize/wire.py src/optio_optimize/batch_backends.py tests/optimize/test_wire.py
git commit -m "One Anthropic response translator, not two

The adapter needs exactly what batch_backends already had. Two copies of a
translation is the specific shape of the bug that made a whole benchmark run
measure nothing when tools went unsent from one of two sites.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `invariants.py` — absolute rules

Rules true of any single request. The provider enforces every one of these, so a violation is a rejected call rather than a quality regression.

**Files:**
- Create: `src/optio_optimize/invariants.py`
- Test: `tests/optimize/test_invariants.py`

**Interfaces:**
- Consumes: `optio_optimize.types.LLMRequest`, `Message`.
- Produces:
  - `Violation` — frozen dataclass with fields `rule: str`, `message_index: int | None`, `role: str | None`.
  - `check(request: LLMRequest) -> tuple[Violation, ...]`
  - Rule name constants: `TOOL_RESULT_UNPAIRED`, `TOOL_RESULT_UNMATCHED_ID`, `EMPTY_CONTENT_UNATTACHED`, `NO_ANSWERABLE_MESSAGE`.

  Task 3 appends `check_transform` to the same module. Task 6 calls both.

- [ ] **Step 1: Write the failing tests**

Create `tests/optimize/test_invariants.py`:

```python
"""Rules the library must not break, and the two shapes they come in.

Absolute rules are enforced by the provider: violating one is a 400. They are
cheap to state and were never the problem.

Preservation rules (Task 3) are enforced by nobody. That is why a stage could
drop the user's task for weeks while 1,304 tests passed.
"""

from __future__ import annotations

import pytest

from optio_optimize import LLMRequest, Message
from optio_optimize.invariants import (
    EMPTY_CONTENT_UNATTACHED,
    NO_ANSWERABLE_MESSAGE,
    TOOL_RESULT_UNMATCHED_ID,
    TOOL_RESULT_UNPAIRED,
    check,
)

pytestmark = pytest.mark.optimize


def _request(*messages: Message) -> LLMRequest:
    return LLMRequest(model="gpt-4o", messages=messages, temperature=0.0)


def _calls(*ids: str) -> Message:
    """An assistant message carrying tool calls, as a real agent emits it."""
    return Message(
        role="assistant",
        content="",
        extra={"tool_calls": [{"id": i, "type": "function"} for i in ids]},
    )


def _result(call_id: str, text: str = "ok") -> Message:
    return Message(role="tool", content=text, extra={"tool_call_id": call_id})


def _rules(request: LLMRequest) -> set[str]:
    return {v.rule for v in check(request)}


class TestAWellFormedRequestPasses:
    def test_a_plain_chat(self):
        assert check(_request(
            Message(role="system", content="terse"),
            Message(role="user", content="hi"),
        )) == ()

    def test_a_real_agent_loop(self):
        assert check(_request(
            Message(role="system", content="terse"),
            Message(role="user", content="do the thing"),
            _calls("c1"),
            _result("c1"),
        )) == ()

    def test_parallel_tool_calls(self):
        assert check(_request(
            Message(role="user", content="do two things"),
            _calls("c1", "c2"),
            _result("c1"),
            _result("c2"),
        )) == ()

    def test_a_call_still_awaiting_its_result(self):
        # Mid-loop this is normal: the call has been made, the result has not
        # come back. Only the tool-result-to-call direction is checkable.
        assert check(_request(
            Message(role="user", content="go"),
            _calls("c1"),
        )) == ()


class TestAbsoluteRules:
    def test_a_tool_result_with_no_preceding_call_is_flagged(self):
        request = _request(
            Message(role="system", content="terse"),
            Message(role="user", content="hi"),
            _result("c1"),
        )
        assert TOOL_RESULT_UNPAIRED in _rules(request)

    def test_a_tool_result_whose_id_was_never_called_is_flagged(self):
        request = _request(
            Message(role="user", content="hi"),
            _calls("c1"),
            _result("c9"),
        )
        assert TOOL_RESULT_UNMATCHED_ID in _rules(request)

    def test_empty_content_without_tool_calls_is_flagged(self):
        request = _request(
            Message(role="user", content="hi"),
            Message(role="assistant", content=""),
        )
        assert EMPTY_CONTENT_UNATTACHED in _rules(request)

    def test_empty_content_with_tool_calls_is_fine(self):
        # The single most common real-agent message shape, and the one no
        # fixture in this repo used before 2026-07-29.
        assert check(_request(Message(role="user", content="hi"), _calls("c1"))) == ()

    def test_a_system_only_request_is_flagged(self):
        request = _request(Message(role="system", content="terse"))
        assert NO_ANSWERABLE_MESSAGE in _rules(request)

    def test_an_empty_request_is_flagged(self):
        assert NO_ANSWERABLE_MESSAGE in _rules(_request())


class TestViolationsCarryNoPromptContent:
    def test_no_violation_repeats_message_text(self):
        # Violations are printed and can reach CI logs. Section 10's rule --
        # the same one that makes the fail-open guard log an exception's type
        # and never its message -- applies here.
        secret = "the customer's confidential question"
        request = _request(
            Message(role="system", content=secret),
            Message(role="assistant", content=""),
        )
        rendered = " ".join(f"{v.rule}{v.message_index}{v.role}" for v in check(request))
        assert secret not in rendered
        assert all(secret not in str(v) for v in check(request))

    def test_a_violation_locates_the_problem(self):
        request = _request(Message(role="user", content="hi"), _result("c1"))
        violation = next(v for v in check(request) if v.rule == TOOL_RESULT_UNPAIRED)
        assert violation.message_index == 1
        assert violation.role == "tool"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/optimize/test_invariants.py -q`

Expected: collection error — `ModuleNotFoundError: No module named 'optio_optimize.invariants'`

- [ ] **Step 3: Write `invariants.py`**

Create `src/optio_optimize/invariants.py`:

```python
"""Rules the library must not break, checked outside the request path.

Two rule sets, because the two useful kinds of rule have different shapes.

**Absolute rules** are true of any single request and are enforced by the
provider: violating one is a rejected call. They were never the problem.

**Preservation rules** are true of any *rewrite* and are enforced by nobody.
``trim_history`` dropped the user's task out of agent conversations for weeks;
providers accept a conversation with no user message, so nothing failed, the
model inferred a task from the surrounding tool results, and the wrong answer
was attributed to the model. 1,304 tests missed it because every fixture used
the same tidy alternating-chat shape as the code's own mental model.

The distinction matters because the rule that would have caught it is not
expressible about one request. A caller whose history legitimately begins with
an assistant message is not malformed, and the library is right to send it
unchanged. The library is wrong to *create* one. Only the before-and-after pair
shows the difference.

**This never runs on the request path.** It is per-message work on every call,
for a guarantee the pipeline's fail-open provides another way, and SC-5's
overhead budget is not there to be spent on self-checking. Its two callers are
the test suite and the real-agent probe.

**A violation never carries prompt content** -- only a rule name, an index and
a role. Violations are printed and reach terminal scrollback and CI logs, so
§10's rule applies exactly as it does to the fail-open guard, which logs an
exception's type and never its message.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from optio_optimize.types import LLMRequest, Message

#: A ``tool`` message with no assistant ``tool_calls`` before it. Rejected by
#: every major provider -- the same class of defect that made ``fan_out`` fail
#: all 12 live calls over a missing ``"json"`` literal.
TOOL_RESULT_UNPAIRED = "tool_result_unpaired"

#: A ``tool`` message whose ``tool_call_id`` matches no preceding call.
TOOL_RESULT_UNMATCHED_ID = "tool_result_unmatched_id"

#: An empty message with no ``tool_calls`` attached. An assistant message with
#: empty content *and* a tool call is the most common shape in a real agent
#: loop and is valid; empty and unattached is not.
EMPTY_CONTENT_UNATTACHED = "empty_content_unattached"

#: Nothing but system messages. There is no question to answer.
NO_ANSWERABLE_MESSAGE = "no_answerable_message"


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken rule.

    Attributes:
        rule: One of this module's rule-name constants.
        message_index: Position of the offending message, when one message is
            responsible. ``None`` for rules about the request as a whole.
        role: Role of the offending message, when there is one. Safe to print:
            a role is a fixed vocabulary, never caller text.
    """

    rule: str
    message_index: int | None = None
    role: str | None = None


def check(request: LLMRequest) -> tuple[Violation, ...]:
    """Return every absolute rule ``request`` breaks.

    Args:
        request: The request as it would be sent.

    Returns:
        Violations in message order; empty when the request is well formed.
    """
    messages = request.messages
    violations: list[Violation] = []

    if not any(m.role != "system" for m in messages):
        violations.append(Violation(NO_ANSWERABLE_MESSAGE))

    called_ids: set[str] = set()
    for index, message in enumerate(messages):
        called_ids.update(_call_ids(message))

        if message.role == "tool":
            violations.extend(_check_tool_result(messages, index, message, called_ids))
        elif not message.content and not _call_ids(message):
            violations.append(Violation(EMPTY_CONTENT_UNATTACHED, index, message.role))

    return tuple(violations)


def _check_tool_result(
    messages: Sequence[Message],
    index: int,
    message: Message,
    called_ids: set[str],
) -> list[Violation]:
    """Check one ``tool`` message against the calls that precede it."""
    violations: list[Violation] = []

    # Walk back past any run of sibling tool results to the assistant that
    # issued them: parallel tool calls put several results in a row, and only
    # the first is adjacent to its call.
    cursor = index - 1
    while cursor >= 0 and messages[cursor].role == "tool":
        cursor -= 1
    if cursor < 0 or not _call_ids(messages[cursor]):
        violations.append(Violation(TOOL_RESULT_UNPAIRED, index, message.role))
        return violations

    result_id = message.extra.get("tool_call_id")
    if isinstance(result_id, str) and result_id not in called_ids:
        violations.append(Violation(TOOL_RESULT_UNMATCHED_ID, index, message.role))
    return violations


def _call_ids(message: Message) -> frozenset[str]:
    """Tool-call ids a message carries, in either provider's shape.

    OpenAI puts them in ``extra["tool_calls"]`` as dicts with an ``id``.
    A message with none returns an empty set, which is also how "this is not
    an assistant tool-call message" is expressed.
    """
    raw = message.extra.get("tool_calls")
    if not isinstance(raw, (list, tuple)):
        return frozenset()
    return frozenset(
        str(entry["id"])
        for entry in raw
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    )


def _identity(message: Message) -> tuple[Any, ...]:
    """A hashable stand-in for a message, for set membership across a rewrite.

    Content is included because a stage that rewrites text has changed the
    message; role and name because two messages can share text. Deliberately
    not object identity: stages return new frozen objects rather than mutating,
    so ``is`` comparison finds nothing.
    """
    return (message.role, message.content, message.name)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/optimize/test_invariants.py -q`

Expected: 12 passed

- [ ] **Step 5: Run the gate and commit**

```bash
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check . && .venv/Scripts/python.exe -m mypy && .venv/Scripts/python.exe -m pytest -q
git add src/optio_optimize/invariants.py tests/optimize/test_invariants.py
git commit -m "Absolute request invariants: the rules the provider already enforces

Cheap, uncontroversial, and not where the bug was. They are here because the
preservation rules in the next commit need somewhere to live and because a
checker that only knows about rewrites cannot validate a fixture.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `invariants.py` — preservation rules

The half that matters. Every one of these fails silently in production and gets attributed to the model.

**Files:**
- Modify: `src/optio_optimize/invariants.py` (append)
- Test: `tests/optimize/test_invariants.py` (append)

**Interfaces:**
- Consumes: `Violation`, `_identity`, `_call_ids` from Task 2.
- Produces:
  - `check_transform(original: LLMRequest, sent: LLMRequest) -> tuple[Violation, ...]`
  - Rule constants: `LAST_USER_MESSAGE_DROPPED`, `SYSTEM_PROMPT_DROPPED`, `MESSAGE_ORDER_CHANGED`, `TOOLS_ADDED`, `CALLED_TOOL_REMOVED`.

  Task 6 calls `check_transform`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/optimize/test_invariants.py`:

```python
from dataclasses import replace

from optio_optimize.invariants import (
    CALLED_TOOL_REMOVED,
    LAST_USER_MESSAGE_DROPPED,
    MESSAGE_ORDER_CHANGED,
    SYSTEM_PROMPT_DROPPED,
    TOOLS_ADDED,
    check_transform,
)

_SEARCH = {"type": "function", "function": {"name": "search", "description": "d"}}
_LOOKUP = {"type": "function", "function": {"name": "lookup", "description": "d"}}


def _agent_loop(steps: int = 6) -> LLMRequest:
    """System prompt, one task, then nothing but tool traffic."""
    messages: list[Message] = [
        Message(role="system", content="You are a support agent."),
        Message(role="user", content="Refund the damaged widget."),
    ]
    for step in range(steps):
        messages.append(_calls(f"c{step}"))
        messages.append(_result(f"c{step}"))
    return LLMRequest(model="gpt-4o", messages=tuple(messages), temperature=0.0)


def _transform_rules(original: LLMRequest, sent: LLMRequest) -> set[str]:
    return {v.rule for v in check_transform(original, sent)}


class TestPreservationRules:
    def test_an_untouched_request_passes(self):
        request = _agent_loop()
        assert check_transform(request, request) == ()

    def test_an_ordinary_trim_passes(self):
        # Dropping middle turns is what trim_history is *for*. The rule must
        # not fire on correct behaviour, or it will be switched off.
        original = _agent_loop(steps=6)
        kept = original.messages[:2] + original.messages[-4:]
        assert check_transform(original, original.with_messages(kept)) == ()

    def test_dropping_the_last_user_message_is_flagged(self):
        # The 2026-07-29 defect, exactly: system + tool traffic, no task.
        original = _agent_loop()
        stripped = tuple(m for m in original.messages if m.role != "user")
        assert LAST_USER_MESSAGE_DROPPED in _transform_rules(
            original, original.with_messages(stripped)
        )

    def test_dropping_the_system_prompt_is_flagged(self):
        original = _agent_loop()
        stripped = tuple(m for m in original.messages if m.role != "system")
        assert SYSTEM_PROMPT_DROPPED in _transform_rules(
            original, original.with_messages(stripped)
        )

    def test_reordering_surviving_messages_is_flagged(self):
        original = _agent_loop(steps=2)
        reversed_ = tuple(reversed(original.messages))
        assert MESSAGE_ORDER_CHANGED in _transform_rules(
            original, original.with_messages(reversed_)
        )

    def test_an_inserted_elision_marker_does_not_count_as_reordering(self):
        # TrimHistoryStage inserts a system marker declaring the gap. New
        # messages are allowed; reordering surviving ones is not.
        original = _agent_loop(steps=6)
        marker = Message(role="system", content="[earlier turns omitted]")
        kept = original.messages[:2] + (marker,) + original.messages[-4:]
        assert check_transform(original, original.with_messages(kept)) == ()

    def test_rewriting_a_message_is_not_reordering(self):
        # cap_tool_results truncates content in place. The rewritten message is
        # a new object with new text and must not read as "removed".
        original = _agent_loop(steps=2)
        messages = list(original.messages)
        messages[3] = messages[3].with_content("truncated...")
        assert check_transform(original, original.with_messages(tuple(messages))) == ()

    def test_adding_a_tool_is_flagged(self):
        original = replace(_agent_loop(), tools=(_SEARCH,))
        assert TOOLS_ADDED in _transform_rules(original, replace(original, tools=(_SEARCH, _LOOKUP)))

    def test_pruning_an_uncalled_tool_passes(self):
        original = replace(_agent_loop(steps=0), tools=(_SEARCH, _LOOKUP))
        assert check_transform(original, replace(original, tools=(_SEARCH,))) == ()

    def test_removing_a_tool_the_agent_already_called_is_flagged(self):
        # PruneToolsStage's stated promise, never checked against a real loop.
        original = LLMRequest(
            model="gpt-4o",
            messages=(
                Message(role="user", content="go"),
                Message(
                    role="assistant",
                    content="",
                    extra={"tool_calls": [{"id": "c1", "function": {"name": "lookup"}}]},
                ),
                _result("c1"),
            ),
            tools=(_SEARCH, _LOOKUP),
            temperature=0.0,
        )
        assert CALLED_TOOL_REMOVED in _transform_rules(original, replace(original, tools=(_SEARCH,)))


class TestEveryPreservationRuleCanFail:
    """A rule that cannot fail is not a rule.

    Each case above pairs a damaged transform with the rule it must trigger;
    this asserts the set is complete, so adding a constant without a failing
    case is caught here rather than by nobody.
    """

    def test_all_rule_constants_have_a_failing_case(self):
        from optio_optimize import invariants

        preservation = {
            invariants.LAST_USER_MESSAGE_DROPPED,
            invariants.SYSTEM_PROMPT_DROPPED,
            invariants.MESSAGE_ORDER_CHANGED,
            invariants.TOOLS_ADDED,
            invariants.CALLED_TOOL_REMOVED,
        }
        exercised = {
            name
            for name in dir(invariants)
            if name.isupper() and isinstance(getattr(invariants, name), str)
        }
        covered = {getattr(invariants, n) for n in exercised} & preservation
        assert covered == preservation
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/optimize/test_invariants.py -k Preservation -q`

Expected: FAIL with `ImportError: cannot import name 'CALLED_TOOL_REMOVED'`

- [ ] **Step 3: Append the preservation rules to `invariants.py`**

Add the constants beneath the existing ones:

```python
#: The library removed the caller's last user message. In a chat the first
#: user turn is a stale question; in an agent loop it is *the task*, and every
#: message after it is the agent's own tool traffic. Providers accept a
#: conversation with no user message, so this fails silently: the model infers
#: a task from the evidence and answers a question nobody asked. Measured on
#: 2026-07-29 -- the trimmed arm cost more than the correct one, because a
#: model that has lost the question writes longer.
LAST_USER_MESSAGE_DROPPED = "last_user_message_dropped"

#: The library removed a system message. ``PrefixCacheStage`` depends on the
#: system prompt being present on every call.
SYSTEM_PROMPT_DROPPED = "system_prompt_dropped"

#: Surviving messages came back in a different relative order. No stage claims
#: to reorder messages -- ``reorder_context`` moves blocks *inside* one
#: message's content.
MESSAGE_ORDER_CHANGED = "message_order_changed"

#: The library added a tool the caller did not supply. Stages may remove and
#: minify; inventing a capability is not something any of them claim.
TOOLS_ADDED = "tools_added"

#: The library removed a tool the conversation shows the agent already calling.
#: ``PruneToolsStage``'s stated promise; unverified against a real loop until
#: this rule existed.
CALLED_TOOL_REMOVED = "called_tool_removed"
```

And append the function:

```python
def check_transform(original: LLMRequest, sent: LLMRequest) -> tuple[Violation, ...]:
    """Return every preservation rule broken between ``original`` and ``sent``.

    The rules here cannot be expressed about a single request. A caller whose
    history legitimately opens with an assistant message is not malformed, and
    the library is right to pass it through untouched -- it is wrong to
    *create* one. Only the pair distinguishes those.

    Args:
        original: The request as the caller supplied it.
        sent: The request after every stage ran.

    Returns:
        Violations; empty when the rewrite preserved everything it must.
    """
    violations: list[Violation] = []
    violations.extend(_check_survivors(original, sent))
    violations.extend(_check_order(original, sent))
    violations.extend(_check_tools(original, sent))
    return tuple(violations)


def _check_survivors(original: LLMRequest, sent: LLMRequest) -> list[Violation]:
    """The system prompt and the last user turn must both survive."""
    violations: list[Violation] = []
    sent_identities = {_identity(m) for m in sent.messages}

    last_user = _last_index(original.messages, "user")
    if last_user is not None and _identity(original.messages[last_user]) not in sent_identities:
        violations.append(Violation(LAST_USER_MESSAGE_DROPPED, last_user, "user"))

    for index, message in enumerate(original.messages):
        if message.role == "system" and _identity(message) not in sent_identities:
            violations.append(Violation(SYSTEM_PROMPT_DROPPED, index, "system"))
    return violations


def _check_order(original: LLMRequest, sent: LLMRequest) -> list[Violation]:
    """Messages that survived must appear in their original relative order.

    Only survivors are compared. Inserting a message -- ``TrimHistoryStage``'s
    elision marker, ``StructuredOutputStage``'s system message -- is allowed,
    and so is rewriting one, which makes it a non-survivor under
    :func:`_identity` and simply drops out of the comparison.
    """
    original_order = {_identity(m): i for i, m in enumerate(original.messages)}
    positions = [
        original_order[_identity(m)] for m in sent.messages if _identity(m) in original_order
    ]
    if positions != sorted(positions):
        return [Violation(MESSAGE_ORDER_CHANGED)]
    return []


def _check_tools(original: LLMRequest, sent: LLMRequest) -> list[Violation]:
    """Tools may be removed or minified, never added; called ones must stay."""
    violations: list[Violation] = []
    before, after = _tool_names(original.tools), _tool_names(sent.tools)

    if after - before:
        violations.append(Violation(TOOLS_ADDED))
    for name in _called_tool_names(original.messages) & before:
        if name not in after:
            violations.append(Violation(CALLED_TOOL_REMOVED))
    return violations


def _tool_names(tools: Iterable[dict[str, Any]]) -> frozenset[str]:
    """Tool names, reading both the OpenAI-nested and Anthropic-flat shapes."""
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        source = function if isinstance(function, dict) else tool
        name = source.get("name")
        if isinstance(name, str):
            names.add(name)
    return frozenset(names)


def _called_tool_names(messages: Sequence[Message]) -> frozenset[str]:
    """Names of tools the conversation shows the agent already invoking."""
    names: set[str] = set()
    for message in messages:
        raw = message.extra.get("tool_calls")
        if not isinstance(raw, (list, tuple)):
            continue
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            function = entry.get("function")
            name = function.get("name") if isinstance(function, dict) else entry.get("name")
            if isinstance(name, str):
                names.add(name)
    return frozenset(names)


def _last_index(messages: Sequence[Message], role: str) -> int | None:
    """Index of the last message with ``role``, or ``None``."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == role:
            return index
    return None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/optimize/test_invariants.py -q`

Expected: 23 passed

- [ ] **Step 5: Prove the rule catches the real defect**

Add to `tests/optimize/test_invariants.py`:

```python
class TestAgainstTheRealStage:
    def test_the_trim_history_fix_holds_under_the_rule(self):
        # Not a synthetic transform: the actual stage, on the actual shape that
        # broke. If someone removes the task-anchor floor, this fails.
        from optio_optimize.config import OptimizeConfig
        from optio_optimize.stages.base import StageContext
        from optio_optimize.stages.history import TrimHistoryStage
        from optio_optimize.tokens import HeuristicCounter

        original = _agent_loop(steps=10)
        ctx = StageContext(config=OptimizeConfig(recent_turns=4), counter=HeuristicCounter())
        result = TrimHistoryStage().before(original, ctx)

        assert check_transform(original, result.request) == ()
        assert check(result.request) == ()
```

Run: `.venv/Scripts/python.exe -m pytest tests/optimize/test_invariants.py::TestAgainstTheRealStage -v`

Expected: PASS. If it fails, the rule and the stage disagree — resolve before continuing, because one of them is wrong and this plan assumes it is not the stage.

- [ ] **Step 6: Run the gate and commit**

```bash
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check . && .venv/Scripts/python.exe -m mypy && .venv/Scripts/python.exe -m pytest -q
git add src/optio_optimize/invariants.py tests/optimize/test_invariants.py
git commit -m "Preservation invariants: the rules nobody enforces

The last user message survives, the system prompt survives, surviving messages
keep their order, tools are only ever removed. Every one of these fails
silently in production and gets attributed to the model having a bad day --
which is exactly what happened for the weeks trim_history was dropping the task.

None is expressible about a single request, which is why none was written.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The Anthropic adapter, async

**Files:**
- Create: `src/optio_optimize/adapters/anthropic.py`
- Test: `tests/optimize/test_adapters_anthropic.py`

**Interfaces:**
- Consumes: `wire.anthropic_system_and_turns`, `wire.anthropic_tools`, `wire.response_from_anthropic_message` (Task 1).
- Produces:
  - `wrap_anthropic_client(client, optimizer=None, **overrides)` — mutates and returns the client.
  - Module constants `_RAW = "_raw"`, `_NATIVE = "_native"`, matching `openai_agents.py`.

  Task 5 extends the same function to sync clients. Task 7 uses it for live measurement.

- [ ] **Step 1: Write the failing tests**

Create `tests/optimize/test_adapters_anthropic.py`:

```python
"""wrap_anthropic_client against the real anthropic SDK types, not stand-ins.

Same construction as test_adapters_openai_agents.py: a genuine client with its
HTTP transport mocked, so the request this package builds and the response it
parses are real SDK shapes validated by the SDK's own models. No network, no
API key, no spend.

This adapter matters more than its size suggests. PrefixCacheStage is described
in its own source as the largest lossless saving in the package, and on OpenAI
it contributes exactly zero -- automatic caching lands on both A/B arms, which
is why a simulated 36.3% corrected to -1.8% live. On Anthropic nothing caches
without an explicit cache_control breakpoint, and this adapter is the only path
by which the marker our pipeline places becomes that field.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("anthropic")

import httpx
from anthropic import Anthropic, AsyncAnthropic

from optio_optimize import Optimizer
from optio_optimize.adapters.anthropic import wrap_anthropic_client

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.optimize


class _FakeAnthropic:
    """Records every request body and answers deterministically."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.reply = "hello there"
        self.cache_read = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [{"type": "text", "text": self.reply}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": self.cache_read,
                },
            },
        )


@pytest.fixture
def fake() -> _FakeAnthropic:
    return _FakeAnthropic()


@pytest.fixture
def async_client(fake: _FakeAnthropic) -> Iterator[AsyncAnthropic]:
    transport = httpx.MockTransport(fake.handler)
    client = AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient(transport=transport))
    yield client


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": "claude-haiku-4-5",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(overrides)
    return base


class TestItStillWorks:
    @pytest.mark.asyncio
    async def test_a_real_call_returns_the_providers_own_content(self, async_client, fake):
        wrap_anthropic_client(async_client)
        reply = await async_client.messages.create(**_kwargs())
        assert reply.content[0].text == "hello there"
        assert reply.usage.input_tokens == 100

    @pytest.mark.asyncio
    async def test_the_system_prompt_reaches_the_wire_as_a_block(self, async_client, fake):
        wrap_anthropic_client(async_client)
        await async_client.messages.create(**_kwargs(system="You are terse."))
        body = fake.requests[0]
        assert body["system"] == [{"type": "text", "text": "You are terse."}]

    @pytest.mark.asyncio
    async def test_unmodelled_kwargs_survive_untouched(self, async_client, fake):
        wrap_anthropic_client(async_client)
        await async_client.messages.create(**_kwargs(top_p=0.5, metadata={"user_id": "u1"}))
        body = fake.requests[0]
        assert body["top_p"] == 0.5
        assert body["metadata"] == {"user_id": "u1"}


class TestThePrefixMarkerBecomesCacheControl:
    @pytest.mark.asyncio
    async def test_a_long_system_prompt_gets_a_cache_control_breakpoint(
        self, async_client, fake
    ):
        # The entire reason this adapter exists. PrefixCacheStage only places a
        # marker above MIN_PREFIX_TOKENS (1024), so the prompt must be big.
        wrap_anthropic_client(async_client, prefix_cache=True, exact_cache=False)
        system = "You are a careful assistant. " * 400
        await async_client.messages.create(
            **_kwargs(
                system=system,
                messages=[
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "q2"},
                ],
            )
        )
        blocks = fake.requests[0]["system"]
        assert blocks[0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_a_short_prompt_gets_no_breakpoint(self, async_client, fake):
        # Below the provider's floor a marker is ignored, so placing one would
        # show up in reports as work done for no effect.
        wrap_anthropic_client(async_client, prefix_cache=True, exact_cache=False)
        await async_client.messages.create(**_kwargs(system="short"))
        assert "cache_control" not in fake.requests[0]["system"][0]


class TestStagesReachTheWire:
    @pytest.mark.asyncio
    async def test_a_long_conversation_sends_fewer_messages(self, async_client, fake):
        wrap_anthropic_client(
            async_client, exact_cache=False, prefix_cache=False, trim_history=True, recent_turns=4
        )
        messages = [{"role": "user", "content": "q0"}]
        for turn in range(10):
            messages.append({"role": "assistant", "content": f"a{turn}"})
            messages.append({"role": "user", "content": f"q{turn + 1}"})

        await async_client.messages.create(**_kwargs(messages=messages))

        sent = fake.requests[0]["messages"]
        assert len(sent) < len(messages)
        assert sent[0]["content"] == "q0", "the opening question is the task; never history"

    @pytest.mark.asyncio
    async def test_tools_are_translated_into_anthropic_shape(self, async_client, fake):
        wrap_anthropic_client(async_client, exact_cache=False)
        await async_client.messages.create(
            **_kwargs(
                tools=[{"name": "search", "description": "d", "input_schema": {"type": "object"}}]
            )
        )
        assert fake.requests[0]["tools"][0]["name"] == "search"


class TestCacheHonesty:
    @pytest.mark.asyncio
    async def test_a_second_identical_call_makes_no_real_request(self, async_client, fake):
        wrap_anthropic_client(async_client, exact_cache=True)
        kwargs = _kwargs(temperature=0.0)
        await async_client.messages.create(**kwargs)
        await async_client.messages.create(**kwargs)
        assert len(fake.requests) == 1

    @pytest.mark.asyncio
    async def test_the_cache_hits_usage_is_zeroed_not_the_originals(self, async_client, fake):
        # Returning the stored object would re-bill the original call's usage on
        # every hit, making a cache that saves money look like one that spends
        # it repeatedly. The OpenAI adapter documents the same defect.
        wrap_anthropic_client(async_client, exact_cache=True)
        kwargs = _kwargs(temperature=0.0)
        first = await async_client.messages.create(**kwargs)
        second = await async_client.messages.create(**kwargs)
        assert first.usage.input_tokens == 100
        assert second.usage.input_tokens == 0
        assert second.content[0].text == first.content[0].text


class TestFailOpen:
    @pytest.mark.asyncio
    async def test_untranslatable_kwargs_reach_the_real_client(self, async_client, fake):
        wrap_anthropic_client(async_client)
        # `messages` as a bare string is not something this package can model.
        # It must not raise here; the SDK's own validation owns that.
        with pytest.raises(Exception):
            await async_client.messages.create(model="claude-haiku-4-5", max_tokens=8, messages="x")

    @pytest.mark.asyncio
    async def test_a_streaming_call_bypasses_the_wrapper(self, async_client, fake):
        wrap_anthropic_client(async_client, exact_cache=True)
        optimizer = Optimizer()
        wrap_anthropic_client(async_client, optimizer=optimizer)
        # A stream=True call must not be counted by the optimizer at all.
        try:
            await async_client.messages.create(**_kwargs(stream=True))
        except Exception:
            pass
        assert optimizer.report.requests == 0
```

- [ ] **Step 2: Add the asyncio marker dependency if absent**

Run: `.venv/Scripts/python.exe -c "import pytest_asyncio; print('present')"`

If that fails, add `"pytest-asyncio>=0.23.0"` to the `dev` extra in `pyproject.toml`, add `asyncio_mode = "auto"` under `[tool.pytest.ini_options]`, then `.venv/Scripts/python.exe -m pip install -e ".[dev]"`. If `asyncio_mode = "auto"` is set, delete the `@pytest.mark.asyncio` decorators from the test file above.

Check first — `test_pipeline_async.py` already exists, so the mechanism is likely present:

Run: `.venv/Scripts/python.exe -m pytest tests/optimize/test_pipeline_async.py -q`

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/optimize/test_adapters_anthropic.py -q`

Expected: collection error — `ModuleNotFoundError: No module named 'optio_optimize.adapters.anthropic'`

- [ ] **Step 4: Write the adapter**

Create `src/optio_optimize/adapters/anthropic.py`:

```python
"""Anthropic adapter: wraps a client's ``messages.create``.

**Why this adapter matters more than its size.**
:class:`~optio_optimize.stages.caching.PrefixCacheStage` is described in its own
source as *"the single largest lossless saving available"*, and on OpenAI it
contributes exactly zero -- OpenAI caches any long prefix automatically, for
both arms of an A/B, which is why a simulated 36.3% saving corrected to -1.8%
against the live API. Anthropic caches nothing without an explicit
``cache_control`` breakpoint, and the ``cacheable`` marker this package's
pipeline places is what puts one there. **This adapter is the only path by
which a user reaches that stage's value**; before it, the translation existed
only inside the benchmark module.

**Why the client, not a framework.** Same reasoning as
:mod:`~optio_optimize.adapters.openai_agents`: ``messages.create`` is the one
place anything built on this SDK actually talks to the provider, so wrapping it
composes with the Claude Agent SDK, LangGraph-on-Anthropic, and direct use
alike, without this package importing any of them.

**Sync and async both.** Unlike OpenAI, where the Agents SDK's ``async def``
``get_response`` forced the choice, ``Anthropic`` and ``AsyncAnthropic`` are
both in ordinary use. :class:`~optio_optimize.pipeline.Pipeline` already has
``execute`` and ``aexecute``, so supporting both costs one branch -- and "async
only" is not plug and play.

**Streaming is not optimized.** A stage pipeline built around one request
producing one response can only buffer a token stream, which defeats the reason
to ask for one. A ``stream=True`` call bypasses this wrapper entirely.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, cast

from optio_optimize import wire
from optio_optimize.optimizer import Optimizer
from optio_optimize.types import LLMRequest, LLMResponse, Message

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_log = logging.getLogger("optio_optimize")

#: Scratch key under which the caller's original message param rides through
#: untouched -- multimodal content blocks, provider extensions, anything this
#: package does not model -- restored verbatim unless a stage changed the text.
_RAW = "_raw"

#: Scratch key on a response for the real SDK object, when one exists. Returned
#: verbatim rather than reconstructed, to preserve every field this package
#: does not model (id, stop_sequence, container, ...).
_NATIVE = "_native"


def wrap_anthropic_client(client: Any, optimizer: Optimizer | None = None, **overrides: Any) -> Any:
    """Route ``client.messages.create`` through an ``Optimizer``.

    Mutates and returns the same client, matching the identity contract
    ``optio``'s own adapters use (§6.7) and
    :func:`~optio_optimize.adapters.openai_agents.wrap_openai_client`: callers
    keep the object they built, with one method replaced.

    Works on both ``Anthropic`` and ``AsyncAnthropic``; the coroutine-ness of
    the wrapped method is detected rather than declared.

    Args:
        client: The client to wrap. Non-streaming ``create`` calls are
            optimized from this point on; streaming calls pass through.
        optimizer: Optimizer to route through. Built from ``overrides`` when
            omitted, so ``wrap_anthropic_client(client, exact_cache=False)``
            works without constructing one.
        **overrides: Individual ``OptimizeConfig`` fields. Invalid together
            with ``optimizer``, for the same reason ``Optimizer.__init__``
            rejects the combination.

    Returns:
        ``client``, with ``messages.create`` replaced.
    """
    import inspect

    active = optimizer if optimizer is not None else Optimizer(**overrides)
    original_create = client.messages.create

    if inspect.iscoroutinefunction(original_create):
        client.messages.create = _async_create(original_create, active)
    else:
        client.messages.create = _sync_create(original_create, active)
    return client


def _async_create(
    original: Callable[..., Awaitable[Any]],
    optimizer: Optimizer,
) -> Callable[..., Awaitable[Any]]:
    """Build the replacement for an ``AsyncAnthropic``'s ``messages.create``."""

    async def optimized(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return await original(**kwargs)
        prepared = _translate(kwargs)
        if prepared is None:
            return await original(**kwargs)

        async def call_provider(sent: LLMRequest) -> LLMResponse:
            reply = await original(**_kwargs_from_request(sent, kwargs))
            response = wire.response_from_anthropic_message(reply)
            return _with_native(response, reply)

        return _unwrap(await optimizer.acall(prepared, call_provider), kwargs)

    return optimized


def _sync_create(original: Callable[..., Any], optimizer: Optimizer) -> Callable[..., Any]:
    """Build the replacement for a sync ``Anthropic``'s ``messages.create``."""

    def optimized(**kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return original(**kwargs)
        prepared = _translate(kwargs)
        if prepared is None:
            return original(**kwargs)

        def call_provider(sent: LLMRequest) -> LLMResponse:
            reply = original(**_kwargs_from_request(sent, kwargs))
            response = wire.response_from_anthropic_message(reply)
            return _with_native(response, reply)

        return _unwrap(optimizer.call(prepared, call_provider), kwargs)

    return optimized


def _translate(kwargs: dict[str, Any]) -> LLMRequest | None:
    """Build an :class:`LLMRequest`, or ``None`` to fall back untouched.

    Returning ``None`` rather than raising is the fail-open path (ADR-013 rule
    1). Translation runs *before* any provider call, so falling back here
    cannot double-bill -- unlike a failure after a real call already happened.
    """
    try:
        return _request_from_kwargs(kwargs)
    except Exception as exc:  # noqa: BLE001 - must never break the caller
        # Type only, never the message: an exception payload can carry prompt
        # content, and §10's rule outlives the package boundary.
        _log.warning(
            "optio_optimize: could not translate the request (%s); "
            "calling the provider directly, unoptimized",
            type(exc).__name__,
        )
        return None


def _request_from_kwargs(kwargs: dict[str, Any]) -> LLMRequest:
    """Translate ``messages.create(**kwargs)`` into this package's model.

    Anthropic's ``system`` is a top-level parameter rather than a message role,
    so it is folded into the message list as leading ``system`` messages --
    which is the shape every stage in this package already reasons about, and
    what lets ``PrefixCacheStage`` mark it at all.
    """
    messages: list[Message] = []
    system = kwargs.get("system")
    if isinstance(system, str) and system:
        messages.append(Message(role="system", content=system))
    elif isinstance(system, (list, tuple)):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                messages.append(
                    Message(role="system", content=str(block.get("text", "")), extra={_RAW: block})
                )

    for param in kwargs.get("messages") or ():
        messages.append(_message_from_param(param))

    return LLMRequest(
        model=str(kwargs.get("model") or ""),
        messages=tuple(messages),
        max_tokens=kwargs.get("max_tokens") if isinstance(kwargs.get("max_tokens"), int) else None,
        tools=tuple(kwargs.get("tools") or ()),
        temperature=kwargs["temperature"] if isinstance(kwargs.get("temperature"), (int, float)) else None,
        stop=tuple(kwargs.get("stop_sequences") or ()),
    )


def _message_from_param(param: Any) -> Message:
    """Translate one Anthropic ``MessageParam`` into a :class:`Message`.

    Only ``role`` and text ``content`` are modelled; multimodal content blocks,
    ``tool_use``/``tool_result`` blocks and provider extensions ride through in
    ``extra[_RAW]`` and are restored verbatim unless a stage edits the text.
    """
    if not isinstance(param, dict):
        return Message(role="user", content=str(param))
    content = param.get("content")
    text = content if isinstance(content, str) else ""
    return Message(role=param.get("role", "user"), content=text, extra={_RAW: param})


def _param_from_message(message: Message) -> Any:
    """Translate a possibly-transformed :class:`Message` back to a param.

    The original param is returned untouched when no stage changed the text,
    preserving content blocks and every other field this package does not
    model. Only when the text differs is a new param built, and even then only
    ``content`` changes.
    """
    raw = message.extra.get(_RAW)
    if isinstance(raw, dict):
        if raw.get("content") == message.content:
            return raw
        return {**raw, "content": message.content}
    return {"role": message.role, "content": message.content}


def _kwargs_from_request(sent: LLMRequest, original: dict[str, Any]) -> dict[str, Any]:
    """Rebuild ``create(**kwargs)`` from the request the stages left behind.

    Starts from ``original``, preserving every parameter this package never
    models (``top_p``, ``metadata``, ``extra_headers``, ...), and overrides only
    what a stage could plausibly have changed.
    """
    system_blocks, turns = wire.anthropic_system_and_turns(sent)
    kwargs = dict(original)
    kwargs["model"] = sent.model
    kwargs["messages"] = [
        _param_from_message(m) for m in sent.messages if m.role != "system"
    ]
    if system_blocks:
        kwargs["system"] = system_blocks
    elif "system" in kwargs:
        del kwargs["system"]
    if sent.max_tokens is not None:
        kwargs["max_tokens"] = sent.max_tokens
    tools = wire.anthropic_tools(sent)
    if tools is not None:
        kwargs["tools"] = tools
    if sent.stop:
        kwargs["stop_sequences"] = list(sent.stop)
    return kwargs


def _with_native(response: LLMResponse, native: Any) -> LLMResponse:
    """Attach the SDK's own object so it can be handed back verbatim."""
    from dataclasses import replace

    return replace(response, extra={**response.extra, _NATIVE: native})


def _unwrap(response: LLMResponse, kwargs: dict[str, Any]) -> Any:
    """Return the provider's own object, or a fresh one for a cache hit.

    ``served_from`` is the gate rather than "is a native object available".
    ``ExactCacheStage`` stores the prior response and ``dataclasses.replace``
    preserves ``extra``, so a cache hit *does* carry a native object -- the
    original call's, usage numbers included. Returning it would re-bill a call
    that cost nothing, on every hit. The OpenAI adapter documents the same
    defect, found by its own test suite.
    """
    native = response.extra.get(_NATIVE)
    if response.served_from is None and native is not None:
        return native
    return _message_from_response(response, kwargs)


def _message_from_response(response: LLMResponse, kwargs: dict[str, Any]) -> Any:
    """Build a valid SDK ``Message`` for a response with no real call behind it."""
    from anthropic.types import Message as AnthropicMessage

    return AnthropicMessage.model_validate(
        {
            "id": f"optio-optimize-{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": response.model or kwargs.get("model") or "",
            "content": [{"type": "text", "text": response.content}],
            "stop_reason": response.finish_reason or "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        }
    )
```

Remove the unused `cast` and `time` imports if ruff flags them.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/optimize/test_adapters_anthropic.py -q`

Expected: all pass. If `test_a_long_system_prompt_gets_a_cache_control_breakpoint` fails, check that `PrefixCacheStage._stable_prefix_length` sees the system message — `_request_from_kwargs` must place it first in the message tuple.

- [ ] **Step 6: Run the gate and commit**

```bash
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check . && .venv/Scripts/python.exe -m mypy && .venv/Scripts/lint-imports.exe && .venv/Scripts/python.exe -m pytest -q
git add src/optio_optimize/adapters/anthropic.py tests/optimize/test_adapters_anthropic.py
git commit -m "An Anthropic adapter, because that is where prefix_cache pays

PrefixCacheStage calls itself the largest lossless saving in the package and
contributes exactly zero on OpenAI, where automatic caching lands on both A/B
arms -- the reason a simulated 36.3% corrected to -1.8% live. Anthropic caches
nothing without an explicit breakpoint, and the marker our pipeline places is
what puts one there. Until now the only way to reach that was the benchmark
module.

One line to plug in, sync or async, streaming bypasses, and the translation
routes through wire.py so the two adapters cannot drift into sending different
requests.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Prove the sync client path

Task 4 wrote both branches; this task proves the sync one, which nothing has exercised yet.

**Files:**
- Test: `tests/optimize/test_adapters_anthropic.py` (append)
- Modify: `src/optio_optimize/adapters/anthropic.py` only if a test fails

**Interfaces:**
- Consumes: `wrap_anthropic_client` from Task 4.
- Produces: nothing new.

- [ ] **Step 1: Write the failing tests**

Append to `tests/optimize/test_adapters_anthropic.py`:

```python
@pytest.fixture
def sync_client(fake: _FakeAnthropic) -> Iterator[Anthropic]:
    transport = httpx.MockTransport(fake.handler)
    yield Anthropic(api_key="test", http_client=httpx.Client(transport=transport))


class TestTheSyncClient:
    def test_a_real_call_returns_the_providers_own_content(self, sync_client, fake):
        wrap_anthropic_client(sync_client)
        reply = sync_client.messages.create(**_kwargs())
        assert reply.content[0].text == "hello there"

    def test_stages_reach_the_wire(self, sync_client, fake):
        wrap_anthropic_client(
            sync_client, exact_cache=False, prefix_cache=False, trim_history=True, recent_turns=4
        )
        messages = [{"role": "user", "content": "q0"}]
        for turn in range(10):
            messages.append({"role": "assistant", "content": f"a{turn}"})
            messages.append({"role": "user", "content": f"q{turn + 1}"})

        sync_client.messages.create(**_kwargs(messages=messages))

        sent = fake.requests[0]["messages"]
        assert len(sent) < len(messages)
        assert sent[0]["content"] == "q0"

    def test_a_second_identical_call_makes_no_real_request(self, sync_client, fake):
        wrap_anthropic_client(sync_client, exact_cache=True)
        kwargs = _kwargs(temperature=0.0)
        sync_client.messages.create(**kwargs)
        sync_client.messages.create(**kwargs)
        assert len(fake.requests) == 1

    def test_the_cache_hit_is_a_valid_sdk_object(self, sync_client, fake):
        # The reconstructed object must satisfy the SDK's own model, or a
        # caller reading .usage or .stop_reason on a cache hit gets an
        # AttributeError instead of an answer.
        wrap_anthropic_client(sync_client, exact_cache=True)
        kwargs = _kwargs(temperature=0.0)
        sync_client.messages.create(**kwargs)
        hit = sync_client.messages.create(**kwargs)
        assert hit.stop_reason == "end_turn"
        assert hit.usage.output_tokens == 0
        assert hit.content[0].text == "hello there"

    def test_a_streaming_call_bypasses_the_wrapper(self, sync_client, fake):
        optimizer = Optimizer()
        wrap_anthropic_client(sync_client, optimizer=optimizer)
        try:
            sync_client.messages.create(**_kwargs(stream=True))
        except Exception:
            pass
        assert optimizer.report.requests == 0


class TestBothClientsAgree:
    @pytest.mark.asyncio
    async def test_the_same_request_produces_the_same_wire_body(self, fake):
        # Two branches in one function is two chances to diverge. If they ever
        # send different bodies for the same input, this catches it.
        sync_fake, async_fake = _FakeAnthropic(), _FakeAnthropic()
        sync = Anthropic(
            api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(sync_fake.handler))
        )
        asynchronous = AsyncAnthropic(
            api_key="test",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(async_fake.handler)),
        )
        wrap_anthropic_client(sync, exact_cache=False)
        wrap_anthropic_client(asynchronous, exact_cache=False)

        kwargs = _kwargs(system="You are terse.", stop_sequences=["<END>"])
        sync.messages.create(**kwargs)
        await asynchronous.messages.create(**kwargs)

        assert sync_fake.requests[0] == async_fake.requests[0]
```

- [ ] **Step 2: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/optimize/test_adapters_anthropic.py -q`

Expected: all pass. If `test_the_same_request_produces_the_same_wire_body` fails, the two branches in `wrap_anthropic_client` have diverged — fix by extracting the shared body-building into one helper rather than by adjusting the test.

- [ ] **Step 3: Run the gate and commit**

```bash
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check . && .venv/Scripts/python.exe -m mypy && .venv/Scripts/python.exe -m pytest -q
git add tests/optimize/test_adapters_anthropic.py src/optio_optimize/adapters/anthropic.py
git commit -m "Prove the sync Anthropic path, and that both paths send the same bytes

Two branches in one wrapper is two chances to diverge, and a divergence would
show up as sync and async callers being optimized differently for reasons
nobody could see -- the same failure mode that made batch dispatch share the
stage runner rather than reimplement it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Wire invariants into the probe, and add four scenarios

**Files:**
- Modify: `scripts/real_agent_run.py`
- Test: none — this is a script that spends money and is never run by pytest.

**Interfaces:**
- Consumes: `invariants.check`, `invariants.check_transform` (Tasks 2–3); `optio_optimize.adapters.openai_agents._request_from_kwargs`.
- Produces: nothing other tasks consume.

The probe currently wraps the client twice: `_instrument` records what went out, then `wrap_openai_client` optimizes. To call `check_transform` it needs the request *before* optimization too, so a third layer goes on the outside.

Order after this task: `agent → recorder (original) → optimized_create → counting (sent) → real client`.

- [ ] **Step 1: Add the pairing recorder**

In `scripts/real_agent_run.py`, add to the `Arm` dataclass:

```python
    originals: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
```

Add this function after `_instrument`:

```python
def _record_originals(client: AsyncOpenAI, arm: Arm) -> AsyncOpenAI:
    """Capture each request as the agent built it, before any stage runs.

    Wrapped *outside* the optimizer, where `_instrument` sits inside it. The
    preservation rules need both sides: "a conversation with no user message"
    is not by itself wrong -- a caller may legitimately send one -- and only
    the before-and-after pair shows that the library created it.
    """
    original_create = client.chat.completions.create

    async def recording(**kwargs: Any) -> Any:
        arm.originals.append(kwargs)
        return await original_create(**kwargs)

    client.chat.completions.create = recording  # type: ignore[method-assign]
    return client
```

Change `run_arm` so the recorder goes on last:

```python
    client = AsyncOpenAI()
    _instrument(client, arm)
    if optimizer is not None:
        wrap_openai_client(client, optimizer)
    _record_originals(client, arm)
```

- [ ] **Step 2: Check every rewrite the library performed**

Add after the `Arm` class:

```python
def _check_invariants(arm: Arm) -> None:
    """Run every captured rewrite past the invariant checker.

    Pairs by index: the Nth request the agent made produced the Nth request
    that went out, unless a cache served it, in which case there is no
    outgoing request and nothing to compare.

    Imports the adapter's own kwargs translator rather than writing a second
    one. A second parser is exactly how the two would come to disagree about
    what a request is -- the failure that made a whole benchmark run measure
    nothing when `tools` went unsent from one of two translation sites.
    """
    from optio_optimize.adapters.openai_agents import _request_from_kwargs
    from optio_optimize.invariants import check, check_transform

    for index, (original, sent) in enumerate(zip(arm.originals, arm.bodies)):
        before, after = _request_from_kwargs(original), _request_from_kwargs(sent)
        for violation in (*check(after), *check_transform(before, after)):
            arm.violations.append(f"call {index + 1}: {violation.rule} at message {violation.message_index}")
```

Call it at the end of `run_arm`, just before `return arm`:

```python
    _check_invariants(arm)
    return arm
```

- [ ] **Step 3: Fail the run on a violation**

In `report`, after the per-arm answer printing, add:

```python
    failed = [arm for arm in arms if arm.violations]
    for arm in failed:
        print(f"\n[{arm.label}] INVARIANT VIOLATIONS:")
        for violation in arm.violations:
            print(f"    {violation}")
    if failed:
        raise SystemExit(
            "the library broke a rule it must not break. This is the check that "
            "would have caught the 2026-07-29 trim_history defect on the spot, "
            "rather than after four runs and a wire dump."
        )
```

- [ ] **Step 4: Verify it passes on the current, fixed code**

Run: `.venv/Scripts/python.exe scripts/real_agent_run.py --arm=on`

Expected: the run completes, prints its table, and reports no violations. Cost roughly $0.001.

- [ ] **Step 5: Verify the check can actually fail**

Temporarily set `DEFAULT_ANCHOR_TURNS = 0` behaviour back by editing `src/optio_optimize/stages/history.py` — change `task_anchor = 1 if history and history[0].role == "user" else 0` to `task_anchor = 0`.

Run: `.venv/Scripts/python.exe scripts/real_agent_run.py --arm=on`

Expected: exits non-zero, reporting `last_user_message_dropped`. **Then revert the edit** with `git checkout src/optio_optimize/stages/history.py` and re-run to confirm it passes again.

A check that has never been seen to fail is not known to work.

- [ ] **Step 6: Add the four scenarios**

Replace the single `TASK` constant with a scenario list, and take a `--scenario` argument. Add after `INSTRUCTIONS`:

```python
@dataclass(frozen=True)
class Scenario:
    """One agent task, chosen to bend the message list into a new shape."""

    name: str
    task: str
    why: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="support",
        task=TASK,
        why="four tools in sequence; the shape that found the trim_history defect",
    ),
    Scenario(
        name="parallel",
        task=(
            "Check inventory for SKUs WID-100, WID-103 and WID-109 at the same time, "
            "then say which are in stock. Answer in one sentence."
        ),
        why="several tool results in a row, so a cut can land inside the run",
    ),
    Scenario(
        name="empty_result",
        task=(
            "Find orders for nobody@example.com. If there are none, say so plainly "
            "in one sentence and do not invent an order."
        ),
        why="a tool returning an empty payload, which no fixture here has produced",
    ),
    Scenario(
        name="long_loop",
        task=(
            "For each of ORD-5000, ORD-5003, ORD-5006 and ORD-5009, fetch the full "
            "detail and report the order total. Then give the sum. Be terse."
        ),
        why="~15 steps, so trimming and capping both engage repeatedly",
    ),
)
```

Change `run_arm` to take a `Scenario` and use `scenario.task` in place of `TASK`, and give `Arm.label` the scenario name. In `main`, loop over the selected scenarios.

- [ ] **Step 7: Run every scenario**

Run: `.venv/Scripts/python.exe scripts/real_agent_run.py --scenario=all`

Expected: all four complete with no violations. Cost roughly $0.02 total.

Any violation here is a real finding. Record it, fix the library, and add a pytest case that pins it — the graduation path this whole design exists to make routine.

- [ ] **Step 8: Run the gate and commit**

```bash
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check . && .venv/Scripts/python.exe -m mypy && .venv/Scripts/python.exe -m pytest -q
git add scripts/real_agent_run.py
git commit -m "The probe now checks its own output, and has four shapes to check

Pairs each request as the agent built it with the one that went out, and runs
both past the invariant checker. Verified it fails by reintroducing the
trim_history defect and watching it exit non-zero -- a check nobody has seen
fail is not known to work.

Four scenarios: parallel tool results, an empty payload, a fifteen-step loop,
and the original support task.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Measure `prefix_cache` live on Anthropic

The claim under test, from `caching.py`: on an explicit-caching provider the marker is worth *"roughly 30% of total spend on a long conversation"*. It has never been measured. ADR-015 rule 1: evidence must be live.

**Files:**
- Create: `scripts/measure_anthropic_prefix_cache.py`
- Modify: `src/optio_optimize/config.py` (add the real model id to `PRICING`)
- Modify: `docs/optimize-benchmarks.md`
- Modify: `src/optio_optimize/stages/caching.py` (docstring, only if the number contradicts it)

**Interfaces:**
- Consumes: `wrap_anthropic_client` (Task 4).
- Produces: nothing other tasks consume.

- [ ] **Step 1: Add the real model to the pricing table**

`PRICING` keys `claude-haiku-4`, but the real API model id is dated. Without a matching key, every cost figure reports `None` — the absence-is-not-zero rule working correctly and telling you nothing.

In `src/optio_optimize/config.py`, add to `PRICING`:

```python
    # Verified against anthropic.com/pricing on 2026-07-29. This table shares
    # the core's staleness problem and the same mitigation: it is data,
    # auditable against the vendor's page, and overridable.
    "claude-haiku-4-5-20251001": ModelPricing(1.00, 5.00, 0.10),
```

Before committing, open the vendor pricing page and confirm those three numbers. A wrong price here produces a confident wrong cost figure, which is worse than no figure.

- [ ] **Step 2: Write the measurement script**

Create `scripts/measure_anthropic_prefix_cache.py`:

```python
"""Is PrefixCacheStage worth what its docstring claims? Measure it.

`caching.py` calls the prefix marker the difference between a ~90% input
discount and none, worth "roughly 30% of total spend on a long conversation".
That number has never been measured. It is also the stage this package leans on
hardest, and the same class of claim that was already wrong once: modelling
only explicit-style caching credited this library with a 36.3% saving on
`multi_turn_chat` that OpenAI grants unconditionally, and the live run measured
-1.8%.

ADR-015 rule 2: isolated, one stage at a time. Everything is held constant
except `prefix_cache`, and the disabled arm runs first so the server-side cache
bias works against the result this library wants.

Usage:

    python scripts/measure_anthropic_prefix_cache.py

Spends real money, bounded by a cap. Roughly $0.05.
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

MODEL = "claude-haiku-4-5-20251001"

#: Anthropic ignores a cache_control breakpoint below 1024 tokens on most
#: models, so the system prompt must clear that floor for the stage to do
#: anything at all. A measurement below the floor would report "no benefit"
#: and mean "no test".
SYSTEM_PROMPT = (
    "You are a meticulous claims adjuster. Follow these rules exactly. "
) + ("Consider precedent, documentation, and the policy schedule before answering. " * 120)

TURNS = [
    "Is water damage from a burst pipe covered?",
    "What about the resulting mould?",
    "Does the deductible apply once or twice?",
    "How long does the claimant have to file?",
    "What documentation is required?",
    "Summarize your answers in three lines.",
]


def _load_key() -> None:
    """Read the key in-process only; never onto a command line."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = pathlib.Path(__file__).resolve().parent.parent / ".env"
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() in {"ANTHROPIC_API_KEY", "ANTHROPIC_KEY"}:
            os.environ["ANTHROPIC_API_KEY"] = value.strip().strip("\"'")
            return


_load_key()

from anthropic import Anthropic  # noqa: E402

from optio_optimize import Optimizer  # noqa: E402
from optio_optimize.adapters.anthropic import wrap_anthropic_client  # noqa: E402
from optio_optimize.config import PRICING  # noqa: E402
from optio_optimize.savings import _cost  # noqa: E402


def run(prefix_cache: bool) -> tuple[int, int, int, float]:
    """Hold one conversation. Returns (input, cached, output, usd)."""
    client = Anthropic()
    optimizer = Optimizer(
        prefix_cache=prefix_cache,
        exact_cache=False,
        trim_history=False,
        cap_tool_results=False,
        minify_tools=False,
        structured_output=False,
        adaptive_max_tokens=False,
        deduplicate=False,
        prune_retrieval=False,
        detect_unstable_prefix=False,
    )
    wrap_anthropic_client(client, optimizer=optimizer)

    history: list[dict[str, str]] = []
    totals = [0, 0, 0]
    for turn in TURNS:
        history.append({"role": "user", "content": turn})
        reply = client.messages.create(
            model=MODEL, max_tokens=300, system=SYSTEM_PROMPT, messages=history
        )
        text = "".join(b.text for b in reply.content if getattr(b, "type", "") == "text")
        history.append({"role": "assistant", "content": text})
        cached = getattr(reply.usage, "cache_read_input_tokens", 0) or 0
        totals[0] += reply.usage.input_tokens + cached
        totals[1] += cached
        totals[2] += reply.usage.output_tokens

    usd = _cost(PRICING[MODEL], totals[0], totals[2], totals[1])
    return totals[0], totals[1], totals[2], usd


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    # Disabled arm first, so any residual server-side cache favours the
    # baseline rather than the result this library wants.
    off = run(prefix_cache=False)
    on = run(prefix_cache=True)

    print(f"\n{'arm':<10} {'input':>9} {'cached':>9} {'output':>8} {'cost':>10}")
    print("-" * 50)
    for label, (i, c, o, usd) in (("off", off), ("on", on)):
        print(f"{label:<10} {i:>9,} {c:>9,} {o:>8,} ${usd:>9.5f}")
    if off[3]:
        print(f"\ncost reduction from prefix_cache: {(off[3] - on[3]) / off[3]:.1%}")
    print(f"cache-served input: off {off[1]:,}/{off[0]:,}, on {on[1]:,}/{on[0]:,}")
```

- [ ] **Step 3: Run it**

Run: `.venv/Scripts/python.exe scripts/measure_anthropic_prefix_cache.py`

Expected: two arms, twelve calls, roughly $0.05. The `on` arm should show non-zero `cached`; the `off` arm should show zero or near-zero.

If **both** arms show zero cached tokens, the system prompt is under the provider's floor — lengthen it and re-run. That is a broken measurement, not a negative result.

- [ ] **Step 4: Write the number down, whatever it is**

Add a section to `docs/optimize-benchmarks.md`, immediately before `## Batch dispatch`:

```markdown
## `prefix_cache` on Anthropic: the first measurement it has ever had

`PrefixCacheStage` calls itself the largest lossless saving in the package, and
until now that claim rested on a provider's published discount rather than a
run. On OpenAI the stage is worth nothing — automatic caching lands on both A/B
arms — so Anthropic is the only place it can pay, and until the adapter shipped
there was no way for a user to reach it.

Measured live, `claude-haiku-4-5`, six-turn conversation over a system prompt
above the 1024-token floor, `prefix_cache` isolated (ADR-015 rule 2), disabled
arm first:

| arm | input | cache-served | output | cost |
|---|---|---|---|---|
| `prefix_cache` off | *(fill from the run)* | | | |
| `prefix_cache` on | | | | |

*(One paragraph stating what the number means, in the same register as the
`trim_history` and `concision` sections above: what it says, what it does not
say, and what would change the answer.)*
```

Replace the placeholders with the real figures from Step 3 before committing. **The plan cannot supply them** — inventing a measurement is the one thing this document exists to prevent.

- [ ] **Step 5: Correct the docstring if the number contradicts it**

`src/optio_optimize/stages/caching.py` currently asserts the marker is *"worth roughly 30% of total spend on a long conversation"*. If the measurement disagrees, edit that sentence to the measured figure and cite the date and model, exactly as the same file already does for the 36.3% → −1.8% OpenAI correction.

If it agrees, add the citation anyway — a claim with evidence reads differently from a claim without it, and the file should say which this is.

- [ ] **Step 6: Run the gate and commit**

```bash
.venv/Scripts/python.exe -m ruff check . && .venv/Scripts/python.exe -m ruff format --check . && .venv/Scripts/python.exe -m mypy && .venv/Scripts/python.exe -m pytest -q
git add scripts/measure_anthropic_prefix_cache.py src/optio_optimize/config.py docs/optimize-benchmarks.md src/optio_optimize/stages/caching.py
git commit -m "prefix_cache on Anthropic: the first measurement it has ever had

The stage this package leans on hardest, on the only provider where it can pay,
measured in isolation with the disabled arm first so the server-side cache bias
works against us.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Changelog and success-criteria check

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Verify each success criterion from the spec**

Run each and confirm:

1. `wrap_anthropic_client(client)` optimizes in one line, sync and async — `pytest tests/optimize/test_adapters_anthropic.py -q`
2. `prefix_cache` on Anthropic has a live number with date, model and method — `grep -n "prefix_cache. on Anthropic" docs/optimize-benchmarks.md`
3. Reintroducing the `trim_history` defect fails a test without reading output — confirmed in Task 6 Step 5
4. Full gate green — the five commands in Global Constraints
5. No stage added, no config option added — `git diff <base>..HEAD -- src/optio_optimize/config.py` shows only the `PRICING` row

- [ ] **Step 2: Write the changelog entry**

Add under `## [Unreleased]` → `### Added` in `CHANGELOG.md`, above the `BatchOptimizer` entry:

```markdown
- **An Anthropic adapter — `wrap_anthropic_client(client)`, one line, sync or async.** The package
  had exactly one adapter, so "plug and play" was true for the OpenAI ecosystem and false
  everywhere else. It also matters more than reach: `PrefixCacheStage` describes itself as the
  largest lossless saving here and contributes **zero on OpenAI**, where automatic caching lands on
  both arms of an A/B — the reason a simulated 36.3% saving corrected to −1.8% live. Anthropic
  caches nothing without an explicit `cache_control` breakpoint and our marker is what places one,
  so this is the only path by which a user reaches that stage at all. Streaming bypasses untouched;
  translation routes through `optio_optimize.wire`, shared with the OpenAI adapter and batch
  dispatch, so no two of the three can drift into sending different requests.

- **`optio_optimize.invariants`: rules the library must not break, checked outside the request
  path.** Two sets, because the two useful kinds of rule have different shapes. *Absolute* rules
  are true of any single request and the provider already enforces them. *Preservation* rules are
  true of any rewrite and nobody enforces them — the last user message survives, the system prompt
  survives, surviving messages keep their order, a tool the agent already called is never pruned.
  The second set is where the 2026-07-29 `trim_history` defect lived for weeks: the rule that
  catches it cannot be stated about a single request, because a caller may legitimately send a
  conversation with no user message and the library is only wrong to *create* one. Wired into
  pytest and into the real-agent probe, which now fails on a violation instead of needing someone
  to read a wire dump.
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "Changelog: the Anthropic adapter and the invariant checker

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage.** Work item 1 (adapter) → Tasks 4, 5. Item 2 (live measurement) → Task 7. Item 3 (`invariants.py`) → Tasks 2, 3, and wired in Task 6. Item 4 (four scenarios) → Task 6 Step 6. The spec's "response translation moves into `wire.py`" → Task 1. Success criteria → Task 8. No spec requirement is unassigned.

**Type consistency.** `check` and `check_transform` keep the same names and signatures in Tasks 2, 3 and 6. `Violation` fields (`rule`, `message_index`, `role`) are used identically in the tests of Task 2 and the probe of Task 6. `response_from_anthropic_message` is defined in Task 1 and called in Task 4 under that exact name. `_RAW`/`_NATIVE` match `openai_agents.py`'s existing constants.

**Known risks carried into execution.** Task 4's `_request_from_kwargs` folds Anthropic's top-level `system` into leading `system` messages; if `PrefixCacheStage` does not then mark it, Step 5's test fails and the fix is in the folding, not the test. Task 7 depends on a pricing figure that must be verified against the vendor page rather than trusted from this document.

**Scope.** One subsystem — the optimize package's provider reach and its safety floor. No decomposition needed.

