"""Packaging and import-hygiene guarantees.

Two properties are being defended here:

* **Framework extras are never imported at core import time** (Section 4.4). If
  ``import optio`` pulled in langgraph, every user would pay for every
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
        "import sys; import optio; "
        f"leaked=[m for m in {_FRAMEWORK_MODULES!r} if m in sys.modules]; "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", f"core import leaked framework deps: {result.stdout}"


def test_import_is_fast():
    code = "import time; t=time.perf_counter(); import optio; print((time.perf_counter()-t)*1000)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    elapsed_ms = float(result.stdout.strip())
    assert elapsed_ms < 500, f"import took {elapsed_ms:.0f} ms (budget 500 ms)"


#: Stdlib modules that only ``importlib.metadata`` drags in, and that nothing
#: on the core import path needs. Named individually so a failure says which
#: one reappeared rather than just that the import got slower.
_LAZY_ONLY_MODULES = ("importlib.metadata", "zipfile", "email.message")


def test_version_lookup_stays_off_the_import_path():
    """``__version__`` resolves lazily, so metadata machinery is not imported.

    ``importlib.metadata`` measured at 88 ms of a 211 ms import -- 42% of it --
    for a string most programs never read, and it is a library, so every user
    paid it at startup. The 500 ms budget above is far too loose to notice that
    coming back: an eager ``import importlib.metadata`` added anywhere on the
    core path would restore the cost and still pass.
    """
    code = (
        "import sys, optio; "
        f"present=[m for m in {_LAZY_ONLY_MODULES!r} if m in sys.modules]; "
        "print(','.join(present))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "", (
        f"core import pulled in lazy-only modules: {result.stdout.strip()}. "
        "Something imports importlib.metadata eagerly again."
    )


def test_version_still_resolves_and_is_cached():
    """Laziness must not cost correctness: the version still reads, once.

    A lazy attribute that recomputed on every access would trade a one-off
    import cost for a repeated one -- worse for anything logging the version
    per request.
    """
    code = (
        "import sys, optio; "
        "v1 = optio.__version__; "
        "loaded = 'importlib.metadata' in sys.modules; "
        "v2 = optio.__version__; "
        "cached = optio.__dict__.get('__version__'); "
        "print(v1, v1 == v2, loaded, cached == v1)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    version, stable, loaded, cached = result.stdout.split()

    assert version and version != "0.0.0.dev0", f"version did not resolve: {version!r}"
    assert stable == "True", "repeated access returned a different version"
    assert loaded == "True", "metadata should load on first access, not never"
    assert cached == "True", "version was not cached into the module namespace"


def test_lazy_attributes_do_not_swallow_typos():
    """A module ``__getattr__`` must still raise for names that do not exist.

    The failure mode is a ``__getattr__`` that returns something for every
    name: ``optio.instrment`` would then yield a value instead of an
    AttributeError, and the typo surfaces much later as a confusing TypeError.
    """
    import optio

    for missing in ("version", "PackageNotFoundError", "metadata", "instrment"):
        with pytest.raises(AttributeError):
            getattr(optio, missing)


def test_dir_lists_everything_all_advertises():
    """Lazy names stay discoverable.

    PEP 562 laziness hides an attribute from ``dir()`` until it is first
    accessed, which breaks tab-completion and introspection for a name
    ``__all__`` promises.
    """
    import optio

    listed = dir(optio)
    assert set(optio.__all__) <= set(listed), (
        f"advertised but not in dir(): {sorted(set(optio.__all__) - set(listed))}"
    )


def test_package_is_typed():
    import optio

    package_dir = __import__("pathlib").Path(optio.__file__).parent
    assert (package_dir / "py.typed").is_file(), "py.typed marker missing; consumers lose types"


@pytest.mark.parametrize(
    "module",
    [
        "optio.semconv",
        "optio.errors",
        "optio.config",
        "optio.api",
        "optio.runtime.run_context",
        "optio.lanes.base",
        "optio.store.base",
    ],
)
def test_every_module_imports(module):
    __import__(module)
