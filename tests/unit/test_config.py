"""Configuration precedence, validation, and flag defaults (Section 4.3).

Config errors must surface at setup time, loudly. The alternative -- a typo'd env
var silently leaving a lane disabled -- produces a meter that reports nothing and
looks fine, which is the worst outcome for a monitoring tool.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from optio.config import Config, default_config
from optio.errors import OptioConfigError


class TestDefaults:
    def test_documented_flag_defaults(self):
        config = Config()
        assert config.cost_lane is True
        assert config.behavior_lane is True
        assert config.quality_lane is False  # ADR-003
        assert config.store_backend == "memory"  # ADR-005 / SC-1

    def test_config_is_immutable(self):
        with pytest.raises(FrozenInstanceError):
            Config().cost_lane = False  # type: ignore[misc]


class TestValidation:
    @pytest.mark.parametrize("rate", [-0.1, 1.1, 2.0])
    def test_rejects_out_of_range_sample_rate(self, rate):
        with pytest.raises(OptioConfigError, match="quality_sample_rate"):
            Config(quality_sample_rate=rate)

    @pytest.mark.parametrize("rate", [0.0, 0.5, 1.0])
    def test_accepts_boundary_sample_rates(self, rate):
        assert Config(quality_sample_rate=rate).quality_sample_rate == rate

    def test_rejects_non_positive_ttl(self):
        with pytest.raises(OptioConfigError, match="run_ttl_seconds"):
            Config(run_ttl_seconds=0)

    def test_rejects_non_positive_window(self):
        with pytest.raises(OptioConfigError, match="behavior_window_size"):
            Config(behavior_window_size=0)

    def test_rejects_unknown_backend(self):
        with pytest.raises(OptioConfigError, match="store_backend"):
            Config(store_backend="postgres")  # type: ignore[arg-type]

    def test_redis_backend_is_accepted(self):
        # It raised through 0.3.0 because nothing implemented it: the setting
        # was read by no lane, so a distributed deployment would have run
        # in-process while believing it was shared (ADR-005's addendum). ADR-050
        # built the backend, so refusing it is no longer the honest answer.
        config = Config(store_backend="redis", redis_url="redis://localhost:6379")

        assert config.store_backend == "redis"

    def test_redis_without_a_url_is_a_setup_error(self):
        # Configuration naming a backend it cannot reach is a setup error, not a
        # runtime one: fail-open governs the request path, never the wiring
        # (Section 4.2).
        with pytest.raises(OptioConfigError, match="redis_url"):
            Config(store_backend="redis")

    def test_the_error_names_the_working_alternative(self):
        # An error that only says "no" makes the user go read the source.
        with pytest.raises(OptioConfigError, match="store_backend='memory'"):
            Config(store_backend="redis")

    def test_redis_from_the_environment_is_accepted_too(self, monkeypatch):
        # The env path builds the same Config, but it is a separate entry point
        # and a reader should not have to infer that it shares the behaviour.
        monkeypatch.setenv("OPTIO_STORE_BACKEND", "redis")
        monkeypatch.setenv("OPTIO_REDIS_URL", "redis://localhost:6379")

        config = Config.from_env()

        assert config.store_backend == "redis"
        assert config.redis_url == "redis://localhost:6379"

    def test_the_store_timeout_defaults_to_fifty_milliseconds(self):
        # Short on purpose: this sits on the agent's critical path, where a
        # dropped signal is cheaper than a stalled step.
        assert Config().store_timeout_ms == 50

    def test_rejects_a_non_positive_store_timeout(self):
        with pytest.raises(OptioConfigError, match="store_timeout_ms"):
            Config(store_timeout_ms=0)

    def test_reads_the_store_timeout_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPTIO_STORE_TIMEOUT_MS", "250")

        assert Config.from_env().store_timeout_ms == 250


class TestEnvironment:
    def test_reads_flags_from_env(self, monkeypatch):
        monkeypatch.setenv("OPTIO_QUALITY_LANE", "true")
        monkeypatch.setenv("OPTIO_COST_LANE", "false")
        config = Config.from_env()
        assert config.quality_lane is True
        assert config.cost_lane is False

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_spellings(self, monkeypatch, raw):
        monkeypatch.setenv("OPTIO_QUALITY_LANE", raw)
        assert Config.from_env().quality_lane is True

    @pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off"])
    def test_falsy_spellings(self, monkeypatch, raw):
        monkeypatch.setenv("OPTIO_COST_LANE", raw)
        assert Config.from_env().cost_lane is False

    def test_unparseable_bool_raises(self, monkeypatch):
        monkeypatch.setenv("OPTIO_COST_LANE", "maybe")
        with pytest.raises(OptioConfigError, match="not a boolean"):
            Config.from_env()

    def test_unparseable_float_raises(self, monkeypatch):
        monkeypatch.setenv("OPTIO_QUALITY_SAMPLE_RATE", "lots")
        with pytest.raises(OptioConfigError, match="not a number"):
            Config.from_env()

    def test_unknown_backend_from_env_raises(self, monkeypatch):
        monkeypatch.setenv("OPTIO_STORE_BACKEND", "postgres")
        with pytest.raises(OptioConfigError, match="STORE_BACKEND"):
            Config.from_env()

    def test_defaults_apply_when_unset(self, monkeypatch):
        for var in (
            "OPTIO_COST_LANE",
            "OPTIO_BEHAVIOR_LANE",
            "OPTIO_QUALITY_LANE",
            "OPTIO_STORE_BACKEND",
        ):
            monkeypatch.delenv(var, raising=False)
        assert Config.from_env() == Config()

    def test_default_config_helper_matches_from_env(self, monkeypatch):
        monkeypatch.delenv("OPTIO_QUALITY_LANE", raising=False)
        assert default_config() == Config.from_env()


class TestPrecedence:
    def test_explicit_override_beats_env(self, monkeypatch):
        monkeypatch.setenv("OPTIO_QUALITY_LANE", "false")
        config = Config.from_env().merged_with(quality_lane=True)
        assert config.quality_lane is True

    def test_none_overrides_are_ignored(self):
        base = Config(quality_lane=True)
        assert base.merged_with(quality_lane=None).quality_lane is True

    def test_unknown_override_raises(self):
        with pytest.raises(OptioConfigError, match="unknown config option"):
            Config().merged_with(nonexistent=True)

    def test_merge_revalidates(self):
        with pytest.raises(OptioConfigError, match="quality_sample_rate"):
            Config().merged_with(quality_sample_rate=5.0)

    def test_merge_returns_a_new_object(self):
        base = Config()
        assert base.merged_with(quality_lane=True) is not base
        assert base.quality_lane is False
