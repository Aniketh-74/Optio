"""The test fixtures share a server, and must leave the neighbours alone.

``REDIS_URL`` defaults to ``redis://localhost:6379/15``. Port 6379 is the
conventional one, which on a developer's machine is very often *some other
project's* Redis -- so the suite's own reset routine runs against a server it
does not own, on every invocation, unattended.

It used to be ``flushdb()``. That is correct only for as long as the database
number is right, and a database number is a weak thing to put between a test run
and someone else's data: one stale environment variable, one default that moves,
one `docker compose` mapping a different db, and the suite quietly deletes an
application's keys and passes. It was found by running the suite with no
``OPTIO_TEST_REDIS_URL`` set and then looking at what was actually on 6379.

The reset is now scoped to the ``optio:`` namespace instead, which makes the
blast radius structural rather than conditional. This file is what keeps it that
way -- both halves matter, so both are asserted: it must clear what we own, and
it must not touch what we do not.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from optio.store.redis_client import RedisClient
from tests.integration.test_redis_ledger import connect_or_skip, reset_optio_keys

pytestmark = [pytest.mark.integration, pytest.mark.redis]

#: A key belonging to nobody in this project. Named for what it stands in for:
#: the 5,870-key database that was one wrong db number away from being flushed.
_NEIGHBOUR = "some-other-project:session:abc123"


@pytest.fixture
def client() -> Iterator[RedisClient]:
    """A client that cleans up after itself, including its stand-in neighbour."""
    conn = connect_or_skip(timeout_ms=1000)
    reset_optio_keys(conn)
    yield conn
    reset_optio_keys(conn)
    conn._redis.delete(_NEIGHBOUR)
    conn.close()


def test_the_reset_clears_every_key_this_project_owns(client: RedisClient) -> None:
    """Isolation is the point of the reset; a test inheriting another run's
    keys fails in ways that look like logic bugs."""
    for key in ("optio:run:totals", "optio:b:run:steps", "optio:q:run"):
        client._redis.set(key, "1")

    reset_optio_keys(client)

    assert client.count_keys("optio:*") == 0


def test_the_reset_leaves_another_project_alone(client: RedisClient) -> None:
    """The half that `flushdb()` failed.

    Under the old fixture this key did not survive the reset, and nothing in
    the suite would have reported that -- the tests it isolates passed either
    way. Which is the whole problem: a destructive default that is invisible
    while it is correct.
    """
    client._redis.set(_NEIGHBOUR, "a real session belonging to someone else")

    reset_optio_keys(client)

    assert client._redis.get(_NEIGHBOUR) == "a real session belonging to someone else"


def test_the_reset_is_safe_on_an_empty_keyspace(client: RedisClient) -> None:
    """`DEL` with no arguments is an error, and an empty keyspace is the normal
    state at the start of a run."""
    reset_optio_keys(client)
    reset_optio_keys(client)

    assert client.count_keys("optio:*") == 0
