"""End-to-end pipeline tests against real git repositories.

These drive the CLI exactly as the workflows do — same commands, same environment
variables — so what CI runs is what the suite covers. Only the GitHub API is faked.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tokenreport.cli import EXIT_BUDGET, EXIT_ERROR, EXIT_OK, main
from tokenreport.history import History

from .fakes import FakeServer

SRC = str(Path(__file__).resolve().parents[1] / "src")


def git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)}: {result.stderr}")
    return result.stdout.strip()


APP = '''
from pathlib import Path


def collect():
    yield {
        "id": "agent.main.system_prompt",
        "kind": "system_prompt",
        "tier": "resident",
        "group": "main",
        "text": Path("prompt.txt").read_text(),
        "source": "prompt.txt",
        "cache_prefix": True,
    }
    yield {
        "id": "agent.main.tools",
        "kind": "tool_schema",
        "tier": "resident",
        "group": "main",
        "tools": [{"name": "search", "description": "Find things."}],
        "source": "app.py",
    }
    yield {
        "id": "knowledge.doc.guide",
        "kind": "knowledge_doc",
        "tier": "on_demand",
        "group": "knowledge",
        "text": Path("guide.txt").read_text(),
        "source": "guide.txt",
    }
'''

CONFIG = """
entrypoint = "app:collect"
model = "claude-opus-5"

[budgets]
resident_total = {resident_total}
resident_growth_pct = {growth}

[budgets.component]
"agent.main.system_prompt" = {prompt_budget}

[history]
branch = "token-report-data"
path = "history.json"
"""


@pytest.fixture
def project(tmp_path: Path):
    """A repository with a collector, pushed to a bare remote."""
    remote = tmp_path / "remote.git"
    remote.mkdir()
    git("init", "--bare", "--initial-branch=main", ".", cwd=remote)

    work = tmp_path / "work"
    git("clone", str(remote), str(work), cwd=tmp_path)
    git("config", "user.email", "t@example.test", cwd=work)
    git("config", "user.name", "Test", cwd=work)

    (work / "app.py").write_text(APP)
    (work / "prompt.txt").write_text("You are a helpful assistant. " * 20)
    (work / "guide.txt").write_text("Reference material. " * 60)
    (work / "tokenreport.toml").write_text(
        CONFIG.format(resident_total=4000, growth=15, prompt_budget=2000)
    )
    git("add", "-A", cwd=work)
    git("commit", "-m", "initial", cwd=work)
    git("push", "-u", "origin", "main", cwd=work)
    return work, remote


def run_cli(args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> int:
    """Invoke the CLI in-process with a temporary working directory and environment.

    Each real workflow step is its own process, so collector modules must not persist
    between calls: without this reset, a module imported from one test's temp directory
    would be served from sys.modules to the next, and a test that rewrites the collector
    would silently measure the old one.
    """
    previous_cwd = Path.cwd()
    previous_env = dict(os.environ)
    previous_path = list(sys.path)
    previous_modules = set(sys.modules)

    os.chdir(cwd)
    try:
        for key in list(os.environ):
            if key.startswith("GITHUB_") or key == "ANTHROPIC_API_KEY":
                del os.environ[key]
        os.environ["PYTHONPATH"] = SRC
        os.environ.update(env or {})
        return main(args)
    finally:
        os.chdir(previous_cwd)
        os.environ.clear()
        os.environ.update(previous_env)
        # Drop anything the collector import pulled in, and the temp paths it came from.
        for name in set(sys.modules) - previous_modules:
            del sys.modules[name]
        sys.path[:] = previous_path


def collect(work: Path, output: str = "head.json") -> dict:
    assert run_cli(
        ["collect", "--counter", "heuristic", "-o", output], cwd=work
    ) == EXIT_OK
    return json.loads((work / output).read_text())


class TestCollect:
    def test_writes_a_report_with_both_tiers(self, project):
        work, _ = project
        report = collect(work)
        assert report["totals"]["resident"] > 0
        assert report["totals"]["on_demand"] > 0
        assert {c["id"] for c in report["components"]} == {
            "agent.main.system_prompt",
            "agent.main.tools",
            "knowledge.doc.guide",
        }

    def test_records_the_commit_being_measured(self, project):
        work, _ = project
        report = collect(work)
        assert report["commit"] == git("rev-parse", "HEAD", cwd=work)

    def test_the_counter_is_named_and_marked_inexact(self, project):
        work, _ = project
        report = collect(work)
        assert report["counter"]["exact"] is False
        # The specific counter is stamped, because comparisons key on its identity.
        assert report["counter"]["name"] == "heuristic"

    def test_a_broken_collector_exits_with_an_error_not_a_traceback(self, project):
        work, _ = project
        (work / "app.py").write_text("def collect():\n    raise RuntimeError('boom')\n")
        assert run_cli(["collect", "--counter", "heuristic"], cwd=work) == EXIT_ERROR

    def test_a_mis_tiered_component_is_rejected(self, project):
        work, _ = project
        (work / "app.py").write_text(
            'def collect():\n    yield {"id": "a", "tier": "sometimes", "text": "x"}\n'
        )
        assert run_cli(["collect", "--counter", "heuristic"], cwd=work) == EXIT_ERROR


class TestRecordAndBaseline:
    def test_record_pushes_history_and_a_dashboard(self, project):
        work, remote = project
        collect(work)
        assert run_cli(["record", "--head", "head.json"], cwd=work) == EXIT_OK
        files = git("ls-tree", "--name-only", "token-report-data", cwd=remote).split()
        assert sorted(files) == [".nojekyll", "history.json", "index.html"]

    def test_recorded_history_contains_the_commit(self, project):
        work, remote = project
        report = collect(work)
        run_cli(["record", "--head", "head.json"], cwd=work)
        history = History.loads(
            git("show", "token-report-data:history.json", cwd=remote)
        )
        assert history.entries[0].commit == report["commit"]
        assert history.entries[0].resident == report["totals"]["resident"]

    def test_an_inexact_counter_is_still_recorded(self, project):
        # tiktoken, the default, is not exact. Gating history on exactness would leave a
        # typical repository with a permanently empty trend.
        work, remote = project
        collect(work)
        assert run_cli(["record", "--head", "head.json"], cwd=work) == EXIT_OK
        history = History.loads(
            git("show", "token-report-data:history.json", cwd=remote)
        )
        assert history.entries[0].counter == "heuristic"

    def test_record_also_writes_the_dashboard_to_a_directory(self, project):
        work, _ = project
        collect(work)
        assert (
            run_cli(["record", "--head", "head.json", "--output-dir", "site"], cwd=work)
            == EXIT_OK
        )
        assert "<!doctype html>" in (work / "site" / "index.html").read_text()

    def test_record_does_not_touch_the_working_tree(self, project):
        work, _ = project
        collect(work)
        (work / "scratch.txt").write_text("uncommitted work\n")
        run_cli(["record", "--head", "head.json"], cwd=work)
        assert (work / "scratch.txt").read_text() == "uncommitted work\n"
        assert git("rev-parse", "--abbrev-ref", "HEAD", cwd=work) == "main"

    def test_baseline_reads_the_recorded_entry_for_the_merge_base(self, project):
        work, _ = project
        collect(work)
        run_cli(["record", "--head", "head.json"], cwd=work)
        base_commit = git("rev-parse", "HEAD", cwd=work)

        git("checkout", "-b", "feature", cwd=work)
        (work / "prompt.txt").write_text("You are a helpful assistant. " * 30)
        git("commit", "-am", "grow the prompt", cwd=work)

        assert (
            run_cli(
                ["baseline", "-o", "base.json"],
                cwd=work,
                env={"GITHUB_BASE_REF": "main"},
            )
            == EXIT_OK
        )
        baseline = json.loads((work / "base.json").read_text())
        assert baseline["commit"] == base_commit

    def test_baseline_recovers_component_tiers_from_the_head_report(self, project):
        work, _ = project
        collect(work)
        run_cli(["record", "--head", "head.json"], cwd=work)
        git("checkout", "-b", "feature", cwd=work)
        (work / "prompt.txt").write_text("You are helpful. " * 30)
        git("commit", "-am", "change", cwd=work)
        collect(work, "head2.json")

        run_cli(
            ["baseline", "-o", "base.json", "--head", "head2.json"],
            cwd=work,
            env={"GITHUB_BASE_REF": "main"},
        )
        baseline = json.loads((work / "base.json").read_text())
        tiers = {c["id"]: c["tier"] for c in baseline["components"]}
        # Without tier recovery the knowledge doc would be counted as resident and the
        # comparison would report growth in the wrong tier.
        assert tiers["knowledge.doc.guide"] == "on_demand"
        assert tiers["agent.main.system_prompt"] == "resident"

    def test_baseline_is_empty_when_history_does_not_exist(self, project):
        work, _ = project
        assert (
            run_cli(["baseline", "-o", "base.json"], cwd=work,
                    env={"GITHUB_BASE_REF": "main"})
            == EXIT_OK
        )
        assert not (work / "base.json").exists()


class TestReportGate:
    def test_growth_within_budget_passes(self, project):
        work, _ = project
        collect(work, "base.json")
        collect(work, "head.json")
        assert (
            run_cli(["report", "--head", "head.json", "--base", "base.json"], cwd=work)
            == EXIT_OK
        )

    def test_exceeding_a_component_budget_fails_the_job(self, project):
        work, _ = project
        (work / "tokenreport.toml").write_text(
            CONFIG.format(resident_total=100000, growth=100, prompt_budget=10)
        )
        collect(work)
        assert run_cli(["report", "--head", "head.json"], cwd=work) == EXIT_BUDGET

    def test_exceeding_the_resident_ceiling_fails_the_job(self, project):
        work, _ = project
        (work / "tokenreport.toml").write_text(
            CONFIG.format(resident_total=10, growth=100, prompt_budget=100000)
        )
        collect(work)
        assert run_cli(["report", "--head", "head.json"], cwd=work) == EXIT_BUDGET

    def test_no_fail_reports_violations_without_failing(self, project):
        work, _ = project
        (work / "tokenreport.toml").write_text(
            CONFIG.format(resident_total=10, growth=100, prompt_budget=100000)
        )
        collect(work)
        assert (
            run_cli(["report", "--head", "head.json", "--no-fail"], cwd=work) == EXIT_OK
        )

    def test_growth_beyond_the_rate_budget_fails(self, project):
        work, _ = project
        (work / "tokenreport.toml").write_text(
            CONFIG.format(resident_total=100000, growth=5, prompt_budget=100000)
        )
        collect(work, "base.json")
        (work / "prompt.txt").write_text("You are a helpful assistant. " * 60)
        collect(work, "head.json")

        assert (
            run_cli(["report", "--head", "head.json", "--base", "base.json"], cwd=work)
            == EXIT_BUDGET
        )

    def test_writes_the_job_summary_when_running_in_actions(self, project):
        work, _ = project
        collect(work)
        summary = work / "summary.md"
        run_cli(
            ["report", "--head", "head.json"],
            cwd=work,
            env={"GITHUB_STEP_SUMMARY": str(summary)},
        )
        assert "Token Report" in summary.read_text()

    def test_renders_the_comment_to_a_file(self, project):
        work, _ = project
        collect(work)
        run_cli(
            ["report", "--head", "head.json", "--comment-output", "comment.md"], cwd=work
        )
        assert "tokenreport:comment" in (work / "comment.md").read_text()


class TestPostingToGitHub:
    def handler(self):
        state = {"comments": [], "checks": [], "next_id": 1}

        def handler(method, path, body):
            if method == "GET" and "/comments" in path:
                return 200, state["comments"]
            if method == "POST" and "/comments" in path:
                comment = {"id": state["next_id"], "body": body["body"]}
                state["next_id"] += 1
                state["comments"].append(comment)
                return 201, comment
            if method == "PATCH":
                state["comments"][0]["body"] = body["body"]
                return 200, state["comments"][0]
            if method == "POST" and path.endswith("/check-runs"):
                state["checks"].append(body)
                return 201, {"id": 1}
            return 404, {"message": path}

        handler.state = state
        return handler

    def env(self, server, work: Path, *, pr: int | None = 3) -> dict[str, str]:
        event = work / "event.json"
        event.write_text(json.dumps({"pull_request": {"number": pr}} if pr else {}))
        return {
            "GITHUB_API_URL": server.base_url,
            "GITHUB_TOKEN": "test-token",
            "GITHUB_REPOSITORY": "owner/repo",
            "GITHUB_EVENT_PATH": str(event),
            "GITHUB_RUN_ID": "99",
        }

    def test_posts_a_comment_then_updates_it_on_the_next_run(self, project):
        work, _ = project
        collect(work)
        handler = self.handler()
        with FakeServer(handler) as server:
            env = self.env(server, work)
            run_cli(["report", "--head", "head.json", "--post"], cwd=work, env=env)
            run_cli(["report", "--head", "head.json", "--post"], cwd=work, env=env)
        # One comment, edited — not two comments.
        assert len(handler.state["comments"]) == 1

    def test_creates_a_failing_check_when_over_budget(self, project):
        work, _ = project
        (work / "tokenreport.toml").write_text(
            CONFIG.format(resident_total=10, growth=100, prompt_budget=100000)
        )
        collect(work)
        handler = self.handler()
        with FakeServer(handler) as server:
            exit_code = run_cli(
                ["report", "--head", "head.json", "--post", "--check"],
                cwd=work,
                env=self.env(server, work),
            )
        assert exit_code == EXIT_BUDGET
        assert handler.state["checks"][0]["conclusion"] == "failure"

    def test_creates_a_passing_check_when_within_budget(self, project):
        work, _ = project
        collect(work)
        handler = self.handler()
        with FakeServer(handler) as server:
            run_cli(
                ["report", "--head", "head.json", "--post", "--check"],
                cwd=work,
                env=self.env(server, work),
            )
        assert handler.state["checks"][0]["conclusion"] == "success"

    def test_a_push_event_posts_no_comment_but_still_checks(self, project):
        work, _ = project
        collect(work)
        handler = self.handler()
        with FakeServer(handler) as server:
            run_cli(
                ["report", "--head", "head.json", "--post", "--check"],
                cwd=work,
                env=self.env(server, work, pr=None),
            )
        assert handler.state["comments"] == []
        assert len(handler.state["checks"]) == 1


class TestDashboardCommand:
    def test_builds_a_dashboard_from_recorded_history(self, project):
        work, _ = project
        collect(work)
        run_cli(["record", "--head", "head.json"], cwd=work)

        assert (
            run_cli(
                ["dashboard", "--head", "head.json", "--output-dir", "site"], cwd=work
            )
            == EXIT_OK
        )
        page = (work / "site" / "index.html").read_text()
        assert "<!doctype html>" in page
        assert (work / "site" / ".nojekyll").exists()

    def test_builds_an_empty_dashboard_without_history(self, project):
        work, _ = project
        collect(work)
        assert (
            run_cli(
                ["dashboard", "--head", "head.json", "--output-dir", "site"], cwd=work
            )
            == EXIT_OK
        )
        assert "No history recorded yet" in (work / "site" / "index.html").read_text()


class TestFullPipeline:
    def test_two_commits_produce_a_two_point_series(self, project):
        """The whole loop: measure, record, change, measure, record, compare."""
        work, remote = project

        def record_current():
            report = collect(work)
            assert run_cli(["record", "--head", "head.json"], cwd=work) == EXIT_OK
            return report

        first = record_current()
        (work / "prompt.txt").write_text("You are a helpful assistant. " * 40)
        git("commit", "-am", "grow the prompt", cwd=work)
        second = record_current()

        history = History.loads(
            git("show", "token-report-data:history.json", cwd=remote)
        )
        assert len(history.entries) == 2
        assert second["totals"]["resident"] > first["totals"]["resident"]

        # The dashboard now has a real trend to draw.
        run_cli(["dashboard", "--head", "head.json", "--output-dir", "site"], cwd=work)
        page = (work / "site" / "index.html").read_text()
        assert "<polygon" in page
        assert "One data point so far" not in page
