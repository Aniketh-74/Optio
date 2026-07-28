"""CI must never spend real money (audit item 11).

``tests/frameworks/`` isolates real-*framework* tests behind a dedicated CI
job that installs one framework at a time -- the risk there is a heavy or
conflicting dependency tree, not spend, so importorskip plus a REQUIRE gate
(``OPTIO_REQUIRE_FRAMEWORKS``) is enough (see conftest's ``require_provider_sdk``
docstring for the mirror-image reasoning applied to ``openai``/``anthropic``).

``bench/providers.py``'s ``OpenAIProvider``/``AnthropicProvider`` are a
different risk entirely: a real call there costs real money. This module
confirms that risk never reaches CI -- no workflow sets a real provider API
key, and nothing in the pytest suite calls a live provider. The live A/B
benchmark that *does* spend runs only via ``python -m optio_optimize.bench``,
by hand, guarded by ``SpendGuard`` -- never inside a test, never inside CI.
"""

from __future__ import annotations

import re
from pathlib import Path

_ENVIRON_READ = re.compile(r'os\.environ\[\s*["\'](OPENAI_API_KEY|ANTHROPIC_API_KEY)["\']\s*\]')

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"
TESTS_DIR = Path(__file__).resolve().parents[1]


class TestNoWorkflowCanSpend:
    def test_no_workflow_references_a_real_provider_key(self) -> None:
        # A workflow that set OPENAI_API_KEY/ANTHROPIC_API_KEY from a repo
        # secret would turn every push into a billed run the moment anything
        # in the suite constructed a live provider -- there is no guard at
        # the CI layer, only SpendGuard inside a process someone starts by
        # hand. This is the check that would have caught it.
        offenders = []
        for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            text = path.read_text(encoding="utf-8")
            for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
                if key in text:
                    offenders.append(f"{path.name} references {key}")
        assert not offenders, offenders

    def test_no_workflow_invokes_the_live_benchmark_cli(self) -> None:
        # The only thing in this repo that can spend money is
        # `python -m optio_optimize.bench` run with a live flag. It must
        # never appear in a workflow -- if someone wants live numbers, they
        # run it themselves and watch SpendGuard do its job.
        offenders = [
            path.name
            for path in sorted(WORKFLOWS_DIR.glob("*.yml"))
            if "optio_optimize.bench" in path.read_text(encoding="utf-8")
        ]
        assert not offenders, offenders


class TestNoTestCallsALiveProvider:
    def test_no_test_file_sets_a_real_provider_key_from_the_environment(self) -> None:
        # monkeypatch.setenv(..., "test-key") is fine -- it is what the
        # mocked-SDK tests in this package do. os.environ[...] = <real-looking
        # value read from outside the test>, or a bare `os.environ.get(...)`
        # used to gate a live call, would not be. This only checks for the
        # dangerous shape (reading the real env var to actually use it),
        # not the safe one (setting a fake value for a mocked client).
        offenders = [
            f"{path.relative_to(TESTS_DIR.parent)}: {match.group(0)}"
            for path in sorted(TESTS_DIR.rglob("test_*.py"))
            for match in _ENVIRON_READ.finditer(path.read_text(encoding="utf-8"))
        ]
        assert not offenders, offenders
