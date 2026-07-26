"""Packaging and import-hygiene guarantees.

Two properties are being defended here:

* **Framework extras are never imported at core import time** (Section 4.4). If
  ``import agentmeter`` pulled in langgraph, every user would pay for every
  adapter and the dependency tree would stop being clean.
* **Import stays fast** (Section 11: < 500 ms). Slow imports are a silent tax on
  every process that installs us.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: Extras that must not be present in ``sys.modules`` after a core import.
_FRAMEWORK_MODULES = ("langgraph", "langchain_core", "crewai", "openai", "redis")


def test_core_import_pulls_in_no_framework_deps():
    # Run in a clean interpreter: an already-imported module in this process
    # would mask the failure.
    code = (
        "import sys; import agentmeter; "
        f"leaked=[m for m in {_FRAMEWORK_MODULES!r} if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"core import leaked framework deps: {result.stdout}"


def test_import_is_fast():
    code = (
        "import time; t=time.perf_counter(); import agentmeter; print((time.perf_counter()-t)*1000)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    elapsed_ms = float(result.stdout.strip())
    assert elapsed_ms < 500, f"import took {elapsed_ms:.0f} ms (budget 500 ms)"


def test_package_is_typed():
    import agentmeter

    package_dir = __import__("pathlib").Path(agentmeter.__file__).parent
    assert (package_dir / "py.typed").is_file(), "py.typed marker missing; consumers lose types"


@pytest.mark.parametrize(
    "module",
    [
        "agentmeter.semconv",
        "agentmeter.errors",
        "agentmeter.config",
        "agentmeter.api",
        "agentmeter.runtime.run_context",
        "agentmeter.lanes.base",
        "agentmeter.store.base",
    ],
)
def test_every_module_imports(module):
    __import__(module)
