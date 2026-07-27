"""The demo produces the signals it claims to (M4-5 gate, ADR-006).

ADR-006 makes the demo a deliverable rather than documentation, which means it
needs a test. The specific risk is that a demo rots silently: a threshold moves,
a window default changes, and the headline number quietly becomes zero while the
script still exits cleanly and prints something plausible.

``run_demo.main()`` already verifies its own claims and returns non-zero when
they fail, so the smoke test runs the real demo rather than a copy of its logic.
Anything asserted twice would be free to drift apart.

The container path is checked structurally here -- the compose file parses, the
services and files it references exist -- because a Docker daemon is not
available in unit CI. The `demo` job in CI runs the real `docker compose up`.
"""

from __future__ import annotations

import importlib.util
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType

import pytest

from optio import semconv

pytestmark = pytest.mark.demo

DEMO_DIR = Path(__file__).resolve().parents[2] / "examples" / "demo"


def _load(name: str) -> ModuleType:
    """Import a demo module by path.

    The demo lives outside the package (it is an example, not shipped code), so
    it is loaded by file rather than imported by name.

    Args:
        name: Module file stem, e.g. ``"run_demo"``.

    Returns:
        The imported module.
    """
    if str(DEMO_DIR) not in sys.path:
        sys.path.insert(0, str(DEMO_DIR))
    spec = importlib.util.spec_from_file_location(name, DEMO_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TestTheDemoRuns:
    """The end-to-end claim: run it, and it demonstrates what it says."""

    @pytest.fixture(scope="class")
    @classmethod
    def output(cls) -> tuple[int, str]:
        """Run the demo once and capture its exit code and output.

        Class-scoped: the demo takes a couple of seconds and every assertion
        below reads the same run, so running it per-test would be five times
        slower for no additional coverage.
        """
        run_demo = _load("run_demo")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = run_demo.main()
        return code, buffer.getvalue()

    def test_the_demo_verifies_its_own_claims(self, output: tuple[int, str]) -> None:
        # main() returns non-zero when the governed run did not beat the
        # ungoverned one. This single assertion is the substance of the gate.
        code, text = output
        assert code == 0, f"demo failed its own verification:\n{text}"

    def test_the_loop_is_caught(self, output: tuple[int, str]) -> None:
        _, text = output
        assert semconv.LOOP_STATE_LOOPING in text

    def test_the_policy_stopped_the_run(self, output: tuple[int, str]) -> None:
        _, text = output
        assert "stopped by" in text

    def test_a_real_saving_is_shown(self, output: tuple[int, str]) -> None:
        # Guards the rot case directly: a demo that still runs but saves $0.00
        # exits cleanly and looks fine in a screenshot.
        _, text = output
        match = re.search(r"saved \$(\d+\.\d+)", text)
        assert match is not None, f"no saving reported:\n{text}"
        assert float(match.group(1)) > 0, "demo reported a zero saving"

    def test_the_governed_run_is_shorter(self, output: tuple[int, str]) -> None:
        _, text = output
        match = re.search(r"caught the loop after (\d+) steps instead of (\d+)", text)
        assert match is not None, f"no step comparison reported:\n{text}"
        governed, ungoverned = int(match.group(1)), int(match.group(2))
        assert governed < ungoverned


class TestTheDemoRunsOffline:
    """No API keys, no network -- SC-1 and the reason the model is scripted."""

    def test_no_api_key_is_read(self) -> None:
        # A demo that needs a key is a demo nobody runs. Checked against the
        # source so adding one is a visible test failure rather than a
        # discovery on someone else's machine.
        for path in DEMO_DIR.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for secret in ("API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
                assert secret not in text, f"{path.name} references {secret}"

    def test_the_model_is_scripted(self) -> None:
        agent = _load("agent")
        assert hasattr(agent, "ScriptedModel")

    def test_the_pricing_model_is_real(self) -> None:
        # The dollar figures are only meaningful if the model is in the pricing
        # table. A typo here would silently produce a $0.00 demo.
        from optio.lanes.cost.pricing import known_models

        agent = _load("agent")
        assert agent.MODEL in known_models()


class TestPolicyRulesMatchTheShippedPacks:
    """The demo policy must not teach a pattern the real packs forbid."""

    def test_repeating_is_not_gated_on(self) -> None:
        # docs/behavior.md: gate on `looping` and `retry_storm`, alert on
        # `repeating`. A demo that denied on `repeating` would be teaching the
        # false positive that gets a monitoring layer uninstalled.
        policy = _load("policy")
        assert semconv.LOOP_STATE_REPEATING not in policy.BLOCKING_LOOP_STATES
        assert {
            semconv.LOOP_STATE_LOOPING,
            semconv.LOOP_STATE_RETRY_STORM,
        } == policy.BLOCKING_LOOP_STATES

    def test_absent_signals_never_deny(self) -> None:
        # The failure mode every pack is built around. An empty signal set is
        # what a fail-open activation looks like from the policy's side.
        policy = _load("policy")
        assert not policy.evaluate({}).denied

    def test_absent_budget_does_not_deny(self) -> None:
        policy = _load("policy")
        decision = policy.evaluate({semconv.RUN_ACTUAL_COST: 5.0})
        assert not decision.denied, "absence was read as zero headroom"

    def test_a_healthy_run_is_allowed(self) -> None:
        policy = _load("policy")
        decision = policy.evaluate(
            {
                semconv.RUN_LOOP_STATE: semconv.LOOP_STATE_HEALTHY,
                semconv.RUN_PROJECTED_COST: 0.01,
            }
        )
        assert not decision.denied

    def test_looping_is_denied(self) -> None:
        policy = _load("policy")
        decision = policy.evaluate({semconv.RUN_LOOP_STATE: semconv.LOOP_STATE_LOOPING})
        assert decision.denied
        assert decision.reason is not None


class TestTheContainerStackIsWellFormed:
    """Structural checks; the real `docker compose up` runs in the CI demo job."""

    def test_every_referenced_file_exists(self) -> None:
        for name in (
            "docker-compose.yml",
            "Dockerfile",
            "entrypoint.sh",
            "otel-collector-config.yaml",
            "run_demo.py",
            "agent.py",
            "policy.py",
            "README.md",
        ):
            assert (DEMO_DIR / name).is_file(), f"examples/demo/{name} is missing"

    def test_the_entrypoint_has_unix_line_endings(self) -> None:
        # CRLF makes Linux read the shebang as `/bin/sh\r` and the container
        # dies with "no such file or directory", naming neither the file nor
        # the cause. .gitattributes pins this; the test proves it held.
        raw = (DEMO_DIR / "entrypoint.sh").read_bytes()
        assert b"\r\n" not in raw, "entrypoint.sh has CRLF endings and will not run on Linux"

    def test_the_entrypoint_starts_with_a_shebang(self) -> None:
        raw = (DEMO_DIR / "entrypoint.sh").read_bytes()
        assert raw.startswith(b"#!"), "entrypoint.sh has no shebang"

    def test_the_dockerfile_copies_paths_that_exist(self) -> None:
        # The build context is the repo root, so these are repo-relative.
        root = DEMO_DIR.parents[1]
        dockerfile = (DEMO_DIR / "Dockerfile").read_text(encoding="utf-8")
        for match in re.finditer(r"^COPY\s+(.+)$", dockerfile, re.MULTILINE):
            *sources, _destination = match.group(1).split()
            for source in sources:
                assert (root / source.rstrip("/")).exists(), (
                    f"Dockerfile copies {source!r}, which does not exist"
                )

    def test_the_compose_file_wires_the_collector_endpoint(self) -> None:
        compose = (DEMO_DIR / "docker-compose.yml").read_text(encoding="utf-8")
        # Without this the demo runs but exports nothing, and the compose stack
        # silently degrades to what `python run_demo.py` already does.
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" in compose
        assert "collector:4317" in compose
