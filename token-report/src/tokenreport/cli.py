"""Command line interface.

Every workflow step is a single call into this module, so the YAML holds no logic. What
runs in CI is what the test suite runs, and a failure can be reproduced locally with the
same command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import config as config_module
from . import counters as counters_module
from .collect import run as collect_run
from .compare import Severity, compare, evaluate
from .contract import (
    ContractError,
    CountedComponent,
    CounterInfo,
    Kind,
    Report,
    Tier,
)
from .github import (
    GitHubClient,
    GitHubError,
    WorkflowContext,
    budget_annotations,
    check_conclusion,
    check_title,
    write_step_summary,
)
from .gitio import Git, GitError, changed_paths, push_files, resolve_baseline_ref
from .history import History, HistoryError, build_history_files
from .render.dashboard import render_dashboard
from .render.markdown import render_comment, render_summary

EXIT_OK = 0
EXIT_BUDGET = 1
EXIT_ERROR = 2


def _load_config(args: argparse.Namespace) -> config_module.Config:
    path = Path(args.config) if args.config else None
    return config_module.load(path)


def _counter(args: argparse.Namespace, config: config_module.Config):
    return counters_module.resolve(
        model=config.model,
        prefer=args.counter,
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )


def _read_report(path: Path) -> Report:
    try:
        return Report.from_json(json.loads(path.read_text(encoding="utf-8")))
    except OSError as exc:
        raise ContractError(f"could not read report at {path}: {exc}") from exc
    except ValueError as exc:
        raise ContractError(f"{path} is not valid JSON: {exc}") from exc


def cmd_collect(args: argparse.Namespace) -> int:
    config = _load_config(args)
    report = collect_run(config, _counter(args, config))
    payload = json.dumps(report.to_json(), indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(
            f"wrote {args.output}: {report.resident:,} resident, "
            f"{report.on_demand:,} on-demand ({report.counter.name})",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(payload)
    return EXIT_OK


def cmd_baseline(args: argparse.Namespace) -> int:
    """Resolve and emit the baseline report for a pull request.

    Preference order is the recorded history for the merge-base, then a fresh collection
    at that commit. History is preferred because it was measured with the exact counter;
    re-collecting at an old commit requires a checkout the workflow may not have.
    """
    config = _load_config(args)
    context = WorkflowContext.from_env()
    git = Git(root=config.root)

    base_ref = args.base_ref or context.base_ref or context.default_branch
    merge_base = resolve_baseline_ref(git, base_ref=base_ref)
    if merge_base is None:
        print(f"no baseline: could not resolve base ref {base_ref!r}", file=sys.stderr)
        return EXIT_OK

    git.fetch(f"{config.history_branch}:refs/remotes/origin/{config.history_branch}")
    raw = git.show(f"refs/remotes/origin/{config.history_branch}", config.history_path)
    history = History.loads(raw) if raw else History()

    entry = history.find(merge_base)
    if entry is None:
        # Fall back to the newest recorded point: better an approximate reference than
        # none, and the comment says which commit it came from.
        entry = history.latest()
        if entry is not None:
            print(
                f"no history entry for merge-base {merge_base[:7]}; using latest "
                f"recorded commit {entry.commit[:7]}",
                file=sys.stderr,
            )
    if entry is None:
        print("no baseline: history is empty", file=sys.stderr)
        return EXIT_OK

    report = Report(
        model=entry.model,
        counter=CounterInfo(name=entry.counter, exact=True),
        commit=entry.commit,
        generated_at=entry.timestamp,
        tool_set_overhead=entry.tool_set_overhead,
        components=[],
    )
    # Rebuild components from the recorded per-component counts. Tiers are inferred from
    # the head report when available, since history stores counts rather than metadata.
    head = _read_report(Path(args.head)) if args.head else None
    tiers = {
        c.id: (c.tier, c.kind, c.group, c.source, c.cache_prefix)
        for c in (head.components if head else [])
    }

    rebuilt = []
    for cid, tokens in entry.components.items():
        tier, kind, group, source, prefix = tiers.get(
            cid, (Tier.RESIDENT, Kind.OTHER, None, None, False)
        )
        rebuilt.append(
            CountedComponent(
                id=cid, tokens=tokens, tier=tier, kind=kind, group=group,
                source=source, cache_prefix=prefix,
            )
        )
    report.components = rebuilt

    payload = json.dumps(report.to_json(), indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print(f"wrote baseline from {entry.commit[:7]}", file=sys.stderr)
    else:
        sys.stdout.write(payload)
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    """Compare, render every surface, and set the exit status from the budgets."""
    config = _load_config(args)
    context = WorkflowContext.from_env()
    head = _read_report(Path(args.head))
    base = (
        _read_report(Path(args.base))
        if args.base and Path(args.base).is_file()
        else None
    )

    comparison = compare(head, base)
    budgets = evaluate(comparison, config.budgets)

    summary = render_summary(comparison, budgets)
    if args.summary_output:
        Path(args.summary_output).write_text(summary, encoding="utf-8")
    write_step_summary(context, summary)

    comment = render_comment(comparison, budgets, summary_url=context.summary_url)
    if args.comment_output:
        Path(args.comment_output).write_text(comment, encoding="utf-8")

    token = os.environ.get("GITHUB_TOKEN", "")
    if args.post and context.repo and token:
        client = GitHubClient(token=token, repo=context.repo)
        if context.is_pull_request:
            comment_id, created = client.upsert_comment(context.pr_number, comment)
            print(
                f"{'posted' if created else 'updated'} comment {comment_id}",
                file=sys.stderr,
            )
        if args.check and head.commit:
            touched = None
            if base is not None and base.commit:
                touched = changed_paths(Git(root=config.root), base=base.commit)
            client.create_check_run(
                name=args.check_name,
                head_sha=head.commit,
                conclusion=check_conclusion(budgets),
                title=check_title(comparison, budgets),
                summary=summary[:60_000],
                annotations=budget_annotations(budgets, changed_paths=touched),
            )
    elif args.post:
        print(
            "not posting: GITHUB_TOKEN or GITHUB_REPOSITORY is unset "
            "(expected for a local run)",
            file=sys.stderr,
        )

    for note in budgets.skipped:
        print(f"skipped {note}", file=sys.stderr)
    for violation in budgets.violations:
        print(f"budget: {violation.message}", file=sys.stderr)

    if budgets.severity is Severity.FAIL and not args.no_fail:
        return EXIT_BUDGET
    return EXIT_OK


def cmd_record(args: argparse.Namespace) -> int:
    """Append the report to history on the data branch and rebuild the dashboard."""
    config = _load_config(args)
    context = WorkflowContext.from_env()
    head = _read_report(Path(args.head))

    if not head.counter.exact:
        print(
            f"not recording: counter {head.counter.name!r} is approximate, and a trend "
            f"series must not mix measured and estimated points",
            file=sys.stderr,
        )
        return EXIT_OK

    git = Git(root=config.root)
    rendered: dict[str, str] = {}

    def build(parent: str | None) -> dict[str, str]:
        existing = git.show(parent, config.history_path) if parent else None
        history, text = build_history_files(
            existing=existing,
            report=head,
            retention_days=config.retention_days,
            max_entries=config.retention_max_entries,
        )
        files = {config.history_path: text}
        if not args.no_dashboard:
            files["index.html"] = render_dashboard(
                history, head, repo=context.repo
            )
            # A .nojekyll file keeps Pages from ignoring files that start with an
            # underscore, which is a confusing failure to debug after the fact.
            files[".nojekyll"] = ""
        rendered.clear()
        rendered.update(files)
        return files

    if args.dry_run:
        build(
            git.rev_parse(f"refs/remotes/origin/{config.history_branch}")
            if git.fetch(
                f"{config.history_branch}:refs/remotes/origin/{config.history_branch}"
            )
            else None
        )
        for path, content in rendered.items():
            print(f"would write {path} ({len(content):,} bytes)", file=sys.stderr)
        if args.output_dir:
            _write_dir(Path(args.output_dir), rendered)
        return EXIT_OK

    result = push_files(
        git,
        branch=config.history_branch,
        build=build,
        message=(
            f"token report: {head.resident:,} resident tokens at "
            f"{(head.commit or '')[:7]}"
        ),
    )
    print(
        f"recorded on {config.history_branch} as {result.committed[:7]} "
        f"(attempt {result.attempts})",
        file=sys.stderr,
    )
    if args.output_dir:
        _write_dir(Path(args.output_dir), rendered)
    return EXIT_OK


def _write_dir(directory: Path, files: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        target = directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Build the dashboard from the recorded history without writing anything back."""
    config = _load_config(args)
    context = WorkflowContext.from_env()
    head = _read_report(Path(args.head))
    git = Git(root=config.root)

    if args.history and Path(args.history).is_file():
        raw = Path(args.history).read_text(encoding="utf-8")
    else:
        git.fetch(f"{config.history_branch}:refs/remotes/origin/{config.history_branch}")
        raw = git.show(f"refs/remotes/origin/{config.history_branch}", config.history_path)

    history = History.loads(raw) if raw else History()
    files = {
        "index.html": render_dashboard(history, head, repo=context.repo),
        ".nojekyll": "",
    }
    if raw:
        files[config.history_path] = raw
    _write_dir(Path(args.output_dir), files)
    print(
        f"wrote dashboard to {args.output_dir} from {len(history.entries)} entries",
        file=sys.stderr,
    )
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tokenreport",
        description="Measure the token cost of the context this repository assembles.",
    )
    parser.add_argument("--config", help="path to tokenreport.toml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="collect and count components")
    collect.add_argument("-o", "--output", help="write the report JSON here")
    collect.add_argument(
        "--counter",
        default="auto",
        choices=["auto", "anthropic", "offline"],
        help="auto uses the API when a key is present and falls back offline",
    )
    collect.set_defaults(func=cmd_collect)

    baseline = subparsers.add_parser(
        "baseline", help="emit the baseline report for the current pull request"
    )
    baseline.add_argument("-o", "--output", help="write the baseline report JSON here")
    baseline.add_argument("--base-ref", help="override the base branch")
    baseline.add_argument("--head", help="head report, used to recover component tiers")
    baseline.set_defaults(func=cmd_baseline)

    report = subparsers.add_parser(
        "report", help="compare, render, optionally post, and gate on budgets"
    )
    report.add_argument("--head", required=True, help="head report JSON")
    report.add_argument("--base", help="baseline report JSON")
    report.add_argument("--comment-output", help="write the rendered comment here")
    report.add_argument("--summary-output", help="write the rendered summary here")
    report.add_argument("--post", action="store_true", help="post to the GitHub API")
    report.add_argument("--check", action="store_true", help="also create a check run")
    report.add_argument("--check-name", default="Token Report")
    report.add_argument(
        "--no-fail",
        action="store_true",
        help="report budget violations without failing the job",
    )
    report.set_defaults(func=cmd_report)

    record = subparsers.add_parser("record", help="append to history on the data branch")
    record.add_argument("--head", required=True, help="report JSON to record")
    record.add_argument("--dry-run", action="store_true", help="build without pushing")
    record.add_argument("--no-dashboard", action="store_true")
    record.add_argument("--output-dir", help="also write the files here")
    record.set_defaults(func=cmd_record)

    dashboard = subparsers.add_parser(
        "dashboard", help="build the dashboard from recorded history"
    )
    dashboard.add_argument("--head", required=True, help="report JSON")
    dashboard.add_argument("--history", help="read history from this file instead of git")
    dashboard.add_argument("--output-dir", default="dashboard")
    dashboard.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ContractError, HistoryError, GitError, GitHubError,
            counters_module.CounterError, config_module.ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
