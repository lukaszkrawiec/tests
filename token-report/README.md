# Token Report

Coverage-report-style CI accounting for the context an agent application assembles.
Measures the token cost of system prompts, tool schemas, and knowledge-base documents,
reports the delta on every pull request, gates on budgets, and keeps a per-commit trend.

See [`DESIGN.md`](DESIGN.md) for the rationale, the prior-art survey, and the decisions
behind the schema.

## The one idea that matters

Context splits into two tiers whose costs differ by orders of magnitude:

| Tier | What it is | Cost |
|---|---|---|
| **Resident** | System prompts, tool schemas, few-shot examples, the knowledge-base *index* | Paid on **every request, forever** |
| **On-demand** | The knowledge-base documents themselves, fetched by tool call | Paid only on requests that retrieve them |

Under progressive disclosure, adding a 5,000-token document to your knowledge base is
nearly free. Adding the one line that describes it in the index is not, because that line
is resident. A tool that sums both flags the harmless change and hides the real one — so
the tiers are reported separately, budgeted separately, and never added together.

## Getting started

```bash
pip install -e .
```

Write a collector. This is the only code you have to write, and it is the reason the
contract is a Python callable rather than a set of file globs: no glob can see that a
preamble, a role prompt, and a rendered index become one resident block.

```python
# myapp/tokenreport.py
def collect():
    for node in my_graph.nodes:
        yield {
            "id": f"agent.{node.name}.system_prompt",  # stable identity for history
            "kind": "system_prompt",
            "tier": "resident",                        # or "on_demand"
            "group": node.name,
            "text": node.compose_system_prompt(),
            "source": f"prompts/{node.name}.md",       # enables inline PR annotations
            "cache_prefix": True,                      # inside the cacheable prefix
        }
        yield {
            "id": f"agent.{node.name}.tools",
            "kind": "tool_schema",
            "tier": "resident",
            "tools": node.tool_schemas,                # counted as tools, not as JSON
        }
```

Point config at it:

```toml
# tokenreport.toml
entrypoint = "myapp.tokenreport:collect"
model = "claude-opus-5"

[budgets]
resident_total = 20000        # absolute ceiling
resident_growth_pct = 10      # fail if resident grows more than 10% vs the base

[budgets.component]
"agent.researcher.system_prompt" = 4000

[history]
branch = "token-report-data"
```

Both a ceiling and a growth rate are worth setting: growth-only lets a prompt creep to
50k in 9% steps, ceiling-only gives no signal until the day it hard-fails.

Then copy `.github/workflows/token-report.yml` and add an `ANTHROPIC_API_KEY` secret.

## Commands

| Command | Purpose |
|---|---|
| `tokenreport collect -o head.json` | Measure this commit |
| `tokenreport baseline -o base.json --head head.json` | Resolve the merge-base baseline from history |
| `tokenreport report --head head.json --base base.json --post --check` | Compare, comment, gate |
| `tokenreport record --head head.json` | Append to history, rebuild the dashboard |
| `tokenreport dashboard --head head.json --output-dir site` | Rebuild the dashboard only |

Every workflow step is one of these, so the YAML holds no logic and any CI failure
reproduces locally with the same command.

## Counting

Counts come from Anthropic's `/v1/messages/count_tokens`, which is **free** (subject to
its own rate limits, independent of your Messages API limits). Two details matter:

- **Counts are marginal.** Each component is measured against a cached baseline request,
  because counting one in isolation would include fixed per-request overhead and inflate
  every number. Per-tool figures are attribution; the residual cost of having tools
  enabled at all is reported separately as `tool_set_overhead` so the totals reconcile.
- **The model is part of the measurement.** Claude 4.7 and later tokenize roughly 30%
  higher than earlier models for identical text. Every report and history entry records
  its model, the dashboard breaks its line at a change rather than drawing a 30% step as
  a regression, and the comparator refuses to print a delta across one.

Without an `ANTHROPIC_API_KEY` — which is every fork pull request, since those get no
secrets — the counter falls back to an offline approximation. Those runs are labelled
approximate and **refused entry into history**, so a trend never mixes measured and
estimated points.

## What you see

- **A sticky pull request comment** — one comment updated in place, headline resident
  delta, per-tier totals, changed components. Truncates under GitHub's 65,536-character
  cap keeping the largest movers.
- **The job summary** — the untruncated breakdown, grouped by agent.
- **A check run** — pass/fail against budgets, so it can be a required check. Component
  violations also become inline annotations on the diff.
- **A trend dashboard** — one self-contained HTML file, no scripts and no external
  requests, so it works from a workflow artifact, from Pages, and under a strict CSP.

Everything works before GitHub Pages is enabled: the dashboard uploads as a workflow
artifact. To publish it, enable Pages on the history branch or add
`actions/upload-pages-artifact` + `actions/deploy-pages`.

## Testing

```bash
python -m pytest -q
```

No network and no credentials required. The Anthropic and GitHub APIs are exercised
against local HTTP fakes so the real request path is covered, and the git plumbing runs
against real repositories in temp directories — including a deterministic version of the
concurrent-merge race, which is the failure that quietly loses data points in
hand-rolled versions of this.

## Limits

- Absolute count accuracy is only as good as the API; the offline fallback is for
  relative signal, not for budgeting.
- `count_tokens` does not apply caching logic, so `cache_prefix` is our bookkeeping of
  what you declared, not something the API confirms.
- The dashboard uses SVG `<title>` tooltips rather than a scripted crosshair, to stay
  script-free. Every value is also in the table view.
