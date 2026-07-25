# Token Report — design document

**Status:** draft for review. No code written yet.
**Goal:** a coverage-report-style CI system that measures the token cost of the context a
LangGraph application assembles — system prompts, tool schemas, knowledge-base docs — and
reports growth per commit inside GitHub.

---

## 1. Problem

In an agent repository, the thing that silently regresses is not test coverage, it is
**context size**. A system prompt gains a paragraph, a tool gains three parameters with long
descriptions, a knowledge-base index gains eight rows — and every request afterwards costs
more, forever, with no test turning red. The growth is invisible in a diff because each
individual change looks small and reasonable.

We want the same feedback loop code coverage has:

- On a pull request: **what did this change do to my context budget?**
- Over time: **what is the shape of the last six months?**
- At the gate: **fail the build when a budget is blown.**

### The measurement that actually matters

A naive version of this sums the tokens of every prompt and every `.md` file and calls it a
number. That number is wrong in a way that makes the tool useless, because it conflates two
costs that differ by orders of magnitude:

| Tier | What it is | Cost |
|---|---|---|
| **Resident** | Text present in *every* request: system prompt, tool schemas, few-shot examples, the knowledge-base *index/manifest* | Paid on every turn, every run, forever |
| **On-demand** | Text fetched only when a tool call asks for it: the knowledge-base documents themselves | Paid only on the runs that retrieve it |

Under a progressive-disclosure design, adding a 5,000-token document to the knowledge base is
nearly free — but adding the one line that describes it in the index is not, because that line
is resident. A tool that reports `knowledge_base: +5,000 tokens` has flagged the harmless
change and hidden the real one.

**So the tiering is the core of the design, not a feature.** Everything else follows from it:
the totals are reported per tier, budgets are set per tier, and the headline number on the PR
comment is *resident* tokens.

---

## 2. Prior art

I searched for an existing tool before designing this. Summary: **the delivery mechanism is a
solved problem with excellent precedent; the measurement is not.** Nothing off the shelf does
Claude-accurate, composition-aware, tiered static token accounting with per-commit history.

### Closest existing tool: Token Guard (LLM Token Limit Checker)

A GitHub Action that counts tokens in files with tiktoken and enforces limits. Explicitly
pitched at "prompt templates, system instructions, or RAG documents." Writes a markdown table
to the job summary with per-file counts and pass/fail. Inputs: glob `patterns`, `max_tokens`,
`token_limit_mode` (`total` or `per_file`), `encoding`.

This is genuinely the same idea, and it covers roughly a quarter of what we want. Its gaps are
exactly our three requirements:

- **Not Claude-accurate.** tiktoken only; its own docs concede "Claude uses its own tokenizer
  that isn't available in tiktoken," and quote 10–20% variance across encodings. Worse for us:
  Claude 4.7+ (and Fable 5 / Mythos 5) use a newer tokenizer producing ~30% more tokens for the
  same text, so a tiktoken proxy is not a small constant error.
- **Per-file only.** "Cannot compose/concatenate prompts — only counts individual files." It
  cannot see that three files plus a template plus a tool list become one system prompt, and it
  has no concept of resident vs on-demand.
- **No history, no deltas.** Job summary only; no PR comment, no check run, no trend.

If per-file tiktoken budgets were the whole ask, we should just install Token Guard. They
aren't, so we build — but we borrow its glob/budget config shape, and `mdtoken` occupies the
same niche if we want a second reference.

### Runtime observability platforms — adjacent, not substitutes

Langfuse, LangSmith, Braintrust, Traceloop, and Helicone all track token usage well, and
Langfuse in particular integrates with LangGraph and reports cost per trace and token usage
over time. But they measure **production traffic after the fact**. They answer "what did we
spend last week," not "what will this pull request cost me." They need a running app, real
requests, and they cannot block a merge. Complementary, not overlapping.

### promptfoo — closest on the assertion side

Has a `token-count` assertion with a `threshold`, plus a `cost` assertion in dollars, and
documented CI/CD integration. But these assert on the tokens consumed by an **eval run** — you
need test cases and provider calls. It is an eval framework, and using it for static repo
accounting means paying for model calls to measure text you already have on disk. Its
per-provider token statistics are useful prior art for report shape.

### Tokalator — closest on the decomposition side

A context-engineering toolkit (VS Code extension, CLI, MCP server, Python econometrics API,
Postgres usage tracker) that decomposes a live token budget into open files / system prompt /
instructions / conversation history / output reservation, across 17 models and 3 providers.
The decomposition-into-components model is very close to ours, and it has a Python API we could
potentially borrow tokenizer plumbing from. But it is aimed at **interactive AI-coding
assistants** — it instruments the developer's editor session, not a repository's own agents in
CI. There is an arXiv paper if we want the methodology.

### The delivery mechanism — strong precedent, worth copying closely

Two ecosystems have already solved "report a metric per commit inside GitHub":

- **`benchmark-action/github-action-benchmark`** — the structural template. Appends each
  default-branch run's data point to a file on a `gh-pages` branch, renders a trend chart from
  it, compares PRs against the previous point, and has `alert-threshold` + `fail-on-alert` for
  regressions. Critically, its `customSmallerIsBetter` mode ingests **arbitrary metrics** as
  `{name, unit, value}` entries — meaning it can host our numbers directly. See §9 for whether
  we should delegate to it.
- **`size-limit-action`** (+ the `bundlesize` / `pkg-size-action` family) — the UX template.
  JavaScript bundle budgets: computes the delta against the base branch, comments the
  comparison on the PR, rejects the PR if the budget is exceeded. This is the exact interaction
  we want, with bytes swapped for tokens. It pairs with
  **`marocchino/sticky-pull-request-comment`**, the standard update-in-place comment action
  (uses a `header` to keep multiple comments independent).

**Verdict: build the measurement, borrow the delivery.** The novel 25% is the tiered,
composition-aware, Claude-accurate collector. The other 75% is a well-trodden path and we
should not reinvent it.

---

## 3. Architecture

```
┌─ your repo ──────────────────────────────────────────────┐
│  tokenreport.toml         ← config: entrypoint, budgets  │
│  myapp/tokenreport.py     ← YOU write: collect()         │
└──────────────────────────┬───────────────────────────────┘
                           │ imports & calls
                  ┌────────▼────────┐
                  │   collector     │  resolves entrypoint, validates
                  └────────┬────────┘
                           │ Component[]  (text or tool schemas, + tier)
                  ┌────────▼────────┐
                  │    counter      │  Anthropic count_tokens (default)
                  │                 │  offline fallback (fork PRs)
                  └────────┬────────┘
                           │
                     report.json     ← versioned, the one artifact everything reads
                           │
      ┌────────────────────┼────────────────────┬──────────────────┐
      ▼                    ▼                    ▼                  ▼
  comparator          budget check         history append      dashboard
  (vs base ref)       (pass/fail)          (data branch)       (HTML+chart)
      │                    │                    │                  │
      ▼                    ▼                    ▼                  ▼
 sticky PR comment    check run          history.json        Pages / artifact
 + job summary        (required check)    (default branch only)
```

Everything downstream reads `report.json`. That keeps the tokenizer, the renderers, and the
history format independently replaceable, and makes every stage testable from a fixture file
with no network and no GitHub.

---

## 4. The contract

You chose the **Python entrypoint** over file globs, which is right: only your code knows how
prompts are composed and which docs are lazy. Config names a callable:

```toml
# tokenreport.toml
entrypoint = "myapp.tokenreport:collect"
model = "claude-opus-5"          # tokenizer is model-specific — part of the measurement
```

You implement `collect()` returning an iterable of components. Nothing is imported from us at
runtime — plain dicts are accepted so the contract adds no dependency to your app:

```python
def collect():
    yield {
        "id": "agent.researcher.system_prompt",   # stable key — this is the tracking identity
        "kind": "system_prompt",                  # system_prompt|tool_schema|knowledge_doc|few_shot|other
        "tier": "resident",                       # resident | on_demand
        "group": "researcher",                    # optional: graph node / agent, for grouping
        "text": build_researcher_prompt(),        # ...or "tools": [ {...schema...} ]
        "source": "prompts/researcher.md",        # optional: enables inline PR annotations
    }
```

Design notes:

- **`id` is the contract, not the file path.** History is keyed on it, so renaming a file does
  not break the trend line, and refactoring three files into one is visible as intended rather
  than as a spurious deletion plus addition.
- **`tools` is a separate field from `text`** because tool schemas must be counted as tool
  schemas — serialized JSON undercounts the real cost of a tool definition.
- **Lazy by generator.** `collect()` yielding lets a large knowledge base stream instead of
  materializing every document in memory.
- **A validation pass** rejects duplicate ids, unknown tiers, and components with neither
  `text` nor `tools`, with a clear error naming the offender. Cheap, and it prevents silently
  mis-tiered components from corrupting history.

### Do we also want globs?

Recommendation: **no, not in v1.** A glob mode cannot know tiers, so it produces exactly the
misleading number §1 warns about. If you want a zero-config on-ramp for a second repo later, it
belongs behind an explicit `[globs]` section that forces a `tier` per pattern.

---

## 5. Counting

**Default: Anthropic `/v1/messages/count_tokens`.** Confirmed free — "Token counting is free to
use but subject to requests per minute rate limits," 2,000 RPM on Start tier / 4,000 Build /
8,000 Scale, and those limits are independent of Messages API limits. It handles system
prompts, tools, images, and PDFs with the same input shape as Messages.

### Marginal counting

Counting a component in isolation includes fixed per-request overhead, which would inflate
every component and make the sum meaningless. So each count is **marginal against a cached
baseline**:

```
baseline      = count_tokens(messages=[{user: "."}])
count(text)   = count_tokens(system=text, messages=[{user: "."}])  - baseline
count(tools)  = count_tokens(tools=tools, messages=[{user: "."}])  - baseline
```

One extra request per run, and the numbers become additive and attributable.

**Known imprecision to document rather than hide:** per-tool marginal counts do not sum exactly
to the cost of the whole tool set, because enabling tools carries a one-time overhead. The
report will therefore carry per-tool figures *for attribution* plus a measured
`tool_set_overhead` line for the difference, so the total stays honest.

### Fallback

Fork PRs do not get repository secrets, so with an API-only counter the workflow simply fails
for outside contributors. The counter sits behind a small interface with an offline
approximation as fallback. Fallback runs are **labelled approximate in the output and refused
entry into history**, so the trend series never mixes exact and estimated points.

### Tokenizer changes break comparability

Claude 4.7+ / Fable 5 / Mythos 5 tokenize ~30% higher for identical text. If `model` changes in
config, every historical number becomes incomparable. Handling: each report and each history
entry records `model` + counter identity; the dashboard **draws a discontinuity marker at the
change and does not connect the line across it**, and the PR comparator refuses to print a
delta across a tokenizer change, printing "baseline measured on a different tokenizer" instead.
Drawing a 30% step change as if the prompt grew would be actively misleading.

---

## 6. Report schema

Versioned from day one, since history files outlive the code that wrote them.

```json
{
  "schema_version": 1,
  "generated_at": "2026-07-25T12:00:00Z",
  "commit": "abc1234", "ref": "refs/pull/42/merge",
  "model": "claude-opus-5",
  "counter": { "name": "anthropic", "exact": true },
  "totals": { "resident": 12480, "on_demand": 98750, "tool_set_overhead": 312 },
  "components": [
    { "id": "agent.researcher.system_prompt", "tokens": 3120, "tier": "resident",
      "kind": "system_prompt", "group": "researcher", "source": "prompts/researcher.md" }
  ]
}
```

`totals.resident` is the headline. `on_demand` is reported but never summed into a single
"total context" figure, because no single request pays both.

---

## 7. Comparison and budgets

Baseline for a PR is the report from the **merge-base**, fetched from the history branch when
present, recomputed on the fly otherwise. Recomputing costs a second collector run but is
always correct and works before any history exists.

```toml
[budgets]
resident_total = 20000     # absolute ceiling
resident_growth_pct = 10   # fail if resident grows >10% vs base

[budgets.component]
"agent.researcher.system_prompt" = 4000
```

Both absolute ceilings and growth rates are needed: growth-only lets a prompt creep to 50k in
9% steps, ceiling-only gives no signal until the day it hard-fails.

Additions and removals are reported explicitly rather than folded into the total, since a new
component and a grown component call for different reviews.

---

## 8. GitHub surfaces

You asked for all three. In priority order:

**Sticky PR comment** — one comment, updated in place via a hidden marker
(`sticky-pull-request-comment`'s `header`, or the same trick directly through the API). Layout:
headline resident delta, per-tier totals, then a table of changed components only, with
unchanged ones in a `<details>` block. Note the hard constraint: **comment bodies cap at 65,536
characters**, so the renderer truncates the component table with a "…and N more, see job
summary" pointer rather than failing the API call.

**Job summary** (`$GITHUB_STEP_SUMMARY`) — the untruncated breakdown: every component, grouped
by agent, sorted by size. Renders as a page on the run. Free, no permissions needed.

**Check run with budgets** — pass/fail so it can be made a required check. Where a component
has a `source`, budget violations also become **inline annotations on the diff**, so an
oversized doc is flagged on the line rather than in a log.

---

## 9. History and the dashboard

Following the `github-action-benchmark` pattern:

| Trigger | Behaviour |
|---|---|
| `pull_request` | Count → compare to merge-base → comment + summary + check. **No writes.** |
| `push` to default branch | Count → append to `history.json` on the `token-report` orphan branch → rebuild and deploy dashboard. |

Writing history only from the default branch is what keeps the series meaningful: PR runs
measure code that may never land.

**Two failure modes I want to handle up front,** because they are what bite hand-rolled
versions of this:

1. **Concurrent merges race on the data branch.** Two merges land within seconds, both read
   `history.json`, both append, one silently wins. Fixed with a `concurrency` group to
   serialize plus fetch–rebase–retry on push.
2. **Unbounded growth.** Thousands of commits produce a history file that is slow to fetch and
   a chart too dense to read. Fixed with retention: full resolution for recent history,
   downsampled beyond that.

### Open decision: build the dashboard or delegate it?

Because `customSmallerIsBetter` accepts arbitrary `{name, unit, value}` metrics, we could emit
that shape and get history, charts, and regression alerts for free.

- **Delegate:** near-zero code for the whole history layer, battle-tested. But its data model
  is flat — `{name, unit, value}` cannot express tier or group, so we lose the resident /
  on-demand distinction that is the entire point, and ~30 components render as ~30 separate
  charts. Its alert semantics are also ratio-vs-previous, not our absolute-ceiling-plus-growth.
- **Build:** a single self-contained HTML page with a stacked area chart over time (resident
  broken down by group), a separate on-demand series, and tokenizer-change markers. Perhaps
  300–400 lines including an inline chart, no external JS.

**Recommendation: build it, and additionally emit the benchmark-action metrics file.** The
dashboard is where the tiering pays off visually, and the extra export is a dozen lines that
keeps the escape hatch open if you'd rather adopt their charts later.

**Dependency on you:** enabling Pages is a repo-settings action I cannot perform. So the
dashboard also builds as a **workflow artifact**, meaning everything works before Pages is
switched on — and `actions/upload-pages-artifact` + `actions/deploy-pages` is the deploy path
once it is.

---

## 10. Questions for you

1. **Prompt caching.** Growth in a cached prefix is far cheaper than growth after the last
   cache breakpoint. Should the report split resident tokens at cache breakpoints? It is the
   single biggest accuracy win available, but needs `collect()` to declare where your
   breakpoints are. (Note: `count_tokens` does not apply caching logic, so this is our
   bookkeeping, not something the API tells us.)
2. **Per-run estimates.** Do you want a modelled "tokens for a typical graph traversal"
   (resident × turns + expected retrievals), or strictly static component sizes? The former is
   closer to real cost and needs assumptions I'd rather you supply than invent.
3. **On-demand worst case.** Worth reporting "if every doc were retrieved" as a ceiling, or is
   the per-doc breakdown enough?
4. **Granularity.** One report per repo, or per graph/agent when there are several?
5. **Retention.** How long at full resolution before downsampling — 90 days? 500 commits?
6. **Where does this live?** I'll develop it here against a LangGraph-shaped fixture, but it
   could be a standalone `pip`-installable package plus a thin composite action, or just a
   directory you copy. The former is more work and much easier to adopt in a second repo.

## 11. Build plan

Sequenced so each step is independently verifiable:

1. Contract + validation + report schema, with a LangGraph-shaped fixture app (agents, composed
   prompts, tool schemas, knowledge base with a manifest) — testable with a fake counter, no
   network.
2. Counter interface, offline fallback, Anthropic counter with baseline caching.
3. Comparator + budget evaluation.
4. Renderers: markdown (comment + summary), then dashboard HTML.
5. History append with the race and retention handling.
6. Workflows wiring it together, running green in this repo.

## 12. Non-goals

Not evaluating prompt *quality*, not optimizing or rewriting prompts, not measuring output
tokens or latency, not tracking production spend (that is Langfuse/LangSmith's job), and not
counting conversation history at runtime — this measures what the repository commits to
sending, which is the part a pull request can actually change.
