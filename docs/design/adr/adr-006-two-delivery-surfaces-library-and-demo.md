# ADR-006 — Two delivery surfaces: library and standalone demo

**Status:** Accepted
**Date:** 2026-07-26
**Related:** ADR-001, SC-1, R-PROD-1, examples/demo/

## Context

ADR-001 makes optio a source of evidence with no enforcement. That is the right architecture
and a genuine presentation problem: **signals are abstract.** A list of attribute names does not
show anyone what the library is for. The value only appears when something *acts* on a signal —
and by design, the acting is somebody else's code.

The two audiences also want different things. A developer integrating optio wants a library:
`pip install`, one line, their spans carry cost. An evaluator — a maintainer deciding whether to
integrate, a hiring manager, anyone deciding if this is real — wants to *see it work* without
reading a docs site or wiring up an agent of their own.

Serving the second audience with documentation alone fails. Documentation asks the reader to
imagine the result.

## Decision

**Two first-class artifacts, both maintained, both gated in CI.**

**1. The library.** `pip install optio`, `instrument(agent)`. This is the product.

**2. A standalone demo** in `examples/demo/`: `docker compose up`, and nothing else. One
command on a clean machine, no API keys, no accounts, no network egress. It runs a scripted
agent that falls into a retrieval loop, twice — ungoverned and governed — and prints the
difference in dollars.

Four properties make the demo honest rather than a sales asset:

- **The model is scripted.** No keys, no spend, no network. A demo that needs an account is a
  demo nobody runs.
- **It exports over real OTLP to a real OTel Collector.** The signals travel the same path they
  would in production, and the collector's log shows them arriving — the viewer does not have to
  trust the summary printed by the process that computed it.
- **It closes the loop.** A real policy in `policy.py` reads the signals and stops the run,
  because ADR-001 means optio alone cannot demonstrate its own value.
- **It verifies itself.** `main()` returns non-zero unless the governed run genuinely beat the
  ungoverned one, and CI runs `docker compose up` and greps the collector log for the signal
  names. A demo that runs but stops demonstrating anything fails the build rather than printing
  `$0.00`.

**The demo is a milestone deliverable with its own smoke test, not documentation.** It is
allowed to fail the build.

## Alternatives

**Library only, with a README example.** Rejected. Copy-pasteable code still asks the reader to
set up an agent and a backend before seeing anything. R-PROD-1 says the adoption risk is
integration feeling heavy; a code block does not address it.

**A hosted demo.** Rejected for now — it needs infrastructure, an uptime commitment, and a spend
budget, all of which are the hosted-plane concerns deferred by ADR-007. Noted as an option if
adoption warrants it.

**A notebook.** Rejected: it proves the code runs in a notebook, not that signals traverse a
real telemetry pipeline. The collector is the point.

## Consequences

**Good**

- An evaluator sees the loop close in one command, with no prerequisites.
- The demo is executable proof of SC-1 rather than a claim about it.
- Because CI runs the real compose stack, the demo cannot rot silently — the usual fate of
  example directories.

**Costs, accepted deliberately**

- **A second artifact to maintain**, including a Dockerfile, a collector config, and a compose
  file, all of which drift if unattended. Mitigated by making CI run them for real.
- **The demo's agent is fake.** It is a scripted loop, not a real model, so it demonstrates the
  signal pipeline rather than detector accuracy on real traffic. The false-positive rate is
  measured separately (§6.4) precisely because the demo cannot speak to it.
- **The numbers are illustrative.** "$1.83 saved" is true of that scripted run and is not a
  benchmark. Stated in the demo's own README.
