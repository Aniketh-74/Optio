"""``optio_optimize``'s public surface, and the one thing it left out.

``optio`` has had a public-API test since 0.1.0 (``tests/unit/test_public_api``)
and ``optio_optimize`` had none, so its ``__all__`` was whatever accumulated.
What accumulated was 27 names including ``AnthropicBatchBackend`` -- a backend
for work that tolerates hours of latency -- and **neither of the two client
wrappers**, which are the entire plug-and-play story of this package.

The consequence is not that the wrappers were unreachable. It is that reaching
them meant ``from optio_optimize.adapters.anthropic import
wrap_anthropic_client``, and a submodule path is exactly what ADR-012 calls
internal and subject to change in a patch release. So the easy path was
documented nowhere and, if a user found it, it was a path they had no promise
about -- while the hard path (hand-translating an SDK call into ``LLMRequest``)
was the one the package docstring taught.

That is ADR-042's shape a second time: the extension point existed and the
public API did not name it.
"""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType

import pytest

import optio_optimize

pytestmark = pytest.mark.optimize


class TestThePlugAndPlayPathIsPublic:
    """The two wrappers are how a caller actually adopts this package."""

    @pytest.mark.parametrize("name", ["wrap_anthropic_client", "wrap_openai_client"])
    def test_the_wrapper_is_importable_from_the_top_level(self, name: str) -> None:
        assert hasattr(optio_optimize, name), (
            f"{name} is the plug-and-play entry point and is not in the public API"
        )

    @pytest.mark.parametrize("name", ["wrap_anthropic_client", "wrap_openai_client"])
    def test_the_wrapper_is_advertised_in_all(self, name: str) -> None:
        """Reachable but unadvertised is the state this test exists to end."""
        assert name in optio_optimize.__all__

    def test_both_vendors_are_offered_on_equal_footing(self) -> None:
        """One vendor exported and the other not would read as a preference.

        It would not be one: both wrappers support sync and async clients and
        both are covered by live-measured benchmarks.
        """
        exported = set(optio_optimize.__all__)
        assert {"wrap_anthropic_client", "wrap_openai_client"} <= exported


class TestTheAdvertisedSurfaceIsCoherent:
    """The checks ``optio`` has had all along, applied to this package too."""

    def test_every_advertised_name_exists(self) -> None:
        for name in optio_optimize.__all__:
            assert hasattr(optio_optimize, name), name

    def test_all_has_no_duplicates(self) -> None:
        assert len(optio_optimize.__all__) == len(set(optio_optimize.__all__))

    def test_every_public_name_is_advertised(self) -> None:
        """An unadvertised public name is a surface users import anyway.

        Once they have, it cannot be changed without breaking them -- so the
        choice is to advertise it deliberately or make it private, never to
        leave it ambiguous.
        """
        ignored = {"annotations"}
        public = {
            name
            for name, value in vars(optio_optimize).items()
            if not name.startswith("_")
            and name not in ignored
            and not isinstance(value, ModuleType)
        }

        assert public - set(optio_optimize.__all__) == set()

    def test_dir_lists_everything_advertised(self) -> None:
        """PEP 562 laziness must not hide a name ``__all__`` promises."""
        listed = dir(optio_optimize)

        assert set(optio_optimize.__all__) <= set(listed), (
            f"advertised but not in dir(): {sorted(set(optio_optimize.__all__) - set(listed))}"
        )


class TestImportingStaysFreeOfTheVendorSDKs:
    def test_neither_sdk_is_imported_by_importing_this_package(self) -> None:
        """Exporting a vendor wrapper must not make that vendor a dependency.

        Both SDKs are optional extras. If putting the wrappers in ``__init__``
        pulled ``anthropic`` or ``openai`` into every import, the convenience
        would have been paid for by every user who wanted neither -- and the
        failure would be a slow import, not an error, so nothing would catch
        it. Checked in a subprocess because this suite has both installed and
        an in-process check would pass on their being imported already.
        """
        probe = (
            "import sys; import optio_optimize; "
            "print(int(any(m == 'anthropic' or m == 'openai' "
            "or m.startswith(('anthropic.', 'openai.')) for m in sys.modules)))"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.strip() == "0", (
            "importing optio_optimize pulled in a vendor SDK; both are optional extras"
        )
