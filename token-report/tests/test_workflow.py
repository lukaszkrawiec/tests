"""Validate the workflow file against the real CLI.

The workflow deliberately contains no logic, which means the only way it can be wrong is
by calling a command that does not exist or passing a flag that does not parse. Actions
cannot be run from the test suite, so this asserts the contract between the YAML and the
argparse parser directly — it catches a renamed command or a typo'd flag, which is what
would otherwise fail on the first real run.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from tokenreport.cli import build_parser

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/token-report.yml"
# GitHub expression templates are resolved at run time; strip them to recover the
# command's shape. The conditional --dry-run flag is covered separately below.
_EXPRESSION = re.compile(r"\$\{\{.*?\}\}", re.DOTALL)


@pytest.fixture(scope="module")
def workflow() -> dict:
    if not WORKFLOW.is_file():
        pytest.skip(f"{WORKFLOW} not present")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def steps(workflow: dict):
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps", []):
            yield job_name, step


def cli_commands(workflow: dict) -> list[tuple[str, list[str]]]:
    found = []
    for job_name, step in steps(workflow):
        run = step.get("run")
        if not run:
            continue
        cleaned = _EXPRESSION.sub("", run)
        for line in cleaned.replace("\n", " ").split("&&"):
            words = shlex.split(line)
            if words and words[0] == "tokenreport":
                found.append((job_name, words[1:]))
    return found


class TestWorkflowStructure:
    def test_has_the_three_jobs(self, workflow):
        assert set(workflow["jobs"]) == {"test", "report", "record"}

    def test_report_runs_only_on_pull_requests(self, workflow):
        assert "pull_request" in workflow["jobs"]["report"]["if"]

    def test_record_never_runs_on_a_pull_request(self, workflow):
        # A pull request measures code that may never land; recording it would
        # contaminate the series.
        assert workflow["jobs"]["record"]["if"] == "github.event_name != 'pull_request'"

    def test_both_reporting_jobs_wait_for_the_tests(self, workflow):
        assert workflow["jobs"]["report"]["needs"] == "test"
        assert workflow["jobs"]["record"]["needs"] == "test"

    def test_report_job_can_comment_and_check_but_not_write_contents(self, workflow):
        permissions = workflow["jobs"]["report"]["permissions"]
        assert permissions["pull-requests"] == "write"
        assert permissions["checks"] == "write"
        assert permissions["contents"] == "read"

    def test_record_job_can_write_contents(self, workflow):
        assert workflow["jobs"]["record"]["permissions"]["contents"] == "write"

    def test_history_writes_are_serialized_by_a_concurrency_group(self, workflow):
        # Two merges landing seconds apart must not contend for the data branch.
        concurrency = workflow["jobs"]["record"]["concurrency"]
        assert concurrency["group"]
        assert concurrency["cancel-in-progress"] is False

    def test_comparison_uses_a_full_clone(self, workflow):
        # A shallow clone does not contain the merge-base, so there is nothing to
        # compare against.
        for job in ("report", "record"):
            checkout = next(
                step
                for job_name, step in steps(workflow)
                if job_name == job
                and step.get("uses", "").startswith("actions/checkout")
            )
            assert checkout["with"]["fetch-depth"] == 0

    def test_the_api_key_is_optional_not_required(self, workflow):
        # Fork pull requests receive no secrets; the run must degrade, not fail.
        collect_steps = [
            step
            for _, step in steps(workflow)
            if step.get("run", "").strip().startswith("tokenreport collect")
        ]
        assert collect_steps
        for step in collect_steps:
            assert "ANTHROPIC_API_KEY" in step["env"]


class TestWorkflowCommandsParse:
    def test_the_workflow_actually_invokes_the_cli(self, workflow):
        assert cli_commands(workflow), "expected tokenreport invocations in the workflow"

    def test_every_invocation_parses(self, workflow):
        parser = build_parser()
        for job_name, argv in cli_commands(workflow):
            try:
                parser.parse_args(argv)
            except SystemExit as exc:  # argparse exits on a bad flag
                raise AssertionError(
                    f"job {job_name!r} runs `tokenreport {' '.join(argv)}`, "
                    f"which the CLI rejects"
                ) from exc

    def test_the_conditional_dry_run_form_also_parses(self, workflow):
        # The record step appends --dry-run via a GitHub expression, which the stripped
        # form above cannot see.
        parser = build_parser()
        record = [argv for job, argv in cli_commands(workflow) if job == "record"]
        assert record
        for argv in record:
            if argv and argv[0] == "record":
                parser.parse_args([*argv, "--dry-run"])

    def test_the_pipeline_order_is_collect_then_baseline_then_report(self, workflow):
        commands = [argv[0] for job, argv in cli_commands(workflow) if job == "report"]
        assert commands == ["collect", "baseline", "report"]

    def test_the_record_job_collects_before_recording(self, workflow):
        commands = [argv[0] for job, argv in cli_commands(workflow) if job == "record"]
        assert commands == ["collect", "record"]

    def test_report_posts_and_creates_a_check(self, workflow):
        argv = next(
            argv for job, argv in cli_commands(workflow)
            if job == "report" and argv[0] == "report"
        )
        assert "--post" in argv and "--check" in argv

    def test_report_does_not_suppress_budget_failures(self, workflow):
        # --no-fail would make the check advisory and it could not gate a merge.
        argv = next(
            argv for job, argv in cli_commands(workflow)
            if job == "report" and argv[0] == "report"
        )
        assert "--no-fail" not in argv
