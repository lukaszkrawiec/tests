import json
from pathlib import Path

import pytest

from tokenreport.compare import compare, evaluate
from tokenreport.config import Budgets
from tokenreport.contract import CountedComponent, CounterInfo, Kind, Report, Tier
from tokenreport.github import (
    COMMENT_PAGE_SIZE,
    MAX_ANNOTATIONS,
    MAX_COMMENT_PAGES,
    GitHubClient,
    GitHubError,
    WorkflowContext,
    budget_annotations,
    check_conclusion,
    check_title,
    write_step_summary,
)
from tokenreport.render.markdown import COMMENT_MARKER

from .fakes import FakeServer


def report(components, *, commit="a" * 40):
    return Report(
        model="claude-opus-5",
        counter=CounterInfo(name="anthropic", exact=True),
        commit=commit,
        components=[
            CountedComponent(
                id=cid,
                tokens=tokens,
                tier=Tier.RESIDENT,
                kind=Kind.OTHER,
                source=f"prompts/{cid}.md",
            )
            for cid, tokens in components.items()
        ],
    )


def github_handler(*, comments=None, failures=None):
    """A stand-in for the handful of endpoints this tool uses."""
    state = {"comments": list(comments or []), "next_id": 100, "checks": []}
    pending = list(failures or [])

    def handler(method, path, body):
        if pending:
            return pending.pop(0), {"message": "transient"}
        if method == "GET" and "/issues/" in path and "/comments" in path:
            return 200, state["comments"]
        if method == "POST" and "/issues/" in path and "/comments" in path:
            comment = {"id": state["next_id"], "body": body["body"]}
            state["next_id"] += 1
            state["comments"].append(comment)
            return 201, comment
        if method == "PATCH" and "/issues/comments/" in path:
            cid = int(path.rsplit("/", 1)[1])
            for comment in state["comments"]:
                if comment["id"] == cid:
                    comment["body"] = body["body"]
                    return 200, comment
            return 404, {"message": "not found"}
        if method == "POST" and path.endswith("/check-runs"):
            run = {"id": 500 + len(state["checks"]), **body}
            state["checks"].append(run)
            return 201, run
        return 404, {"message": f"unhandled {method} {path}"}

    handler.state = state
    return handler


class TestWorkflowContext:
    def test_reads_the_actions_environment(self):
        context = WorkflowContext.from_env(
            {
                "GITHUB_REPOSITORY": "owner/repo",
                "GITHUB_SHA": "abc",
                "GITHUB_BASE_REF": "main",
                "GITHUB_RUN_ID": "42",
                "GITHUB_EVENT_NAME": "pull_request",
            }
        )
        assert context.repo == "owner/repo"
        assert context.base_ref == "main"
        assert context.run_id == "42"

    def test_empty_base_ref_becomes_none(self):
        # GITHUB_BASE_REF is set but empty on push events.
        assert WorkflowContext.from_env({"GITHUB_BASE_REF": ""}).base_ref is None

    def test_pull_request_number_comes_from_the_event_payload(self, tmp_path: Path):
        event = tmp_path / "event.json"
        event.write_text(json.dumps({"pull_request": {"number": 7},
                                     "repository": {"default_branch": "main"}}))
        context = WorkflowContext.from_env({"GITHUB_EVENT_PATH": str(event)})
        assert context.pr_number == 7
        assert context.is_pull_request
        assert context.default_branch == "main"

    def test_push_event_has_no_pull_request_number(self, tmp_path: Path):
        event = tmp_path / "event.json"
        event.write_text(json.dumps({"ref": "refs/heads/main"}))
        context = WorkflowContext.from_env({"GITHUB_EVENT_PATH": str(event)})
        assert context.pr_number is None
        assert not context.is_pull_request

    def test_malformed_event_payload_does_not_crash(self, tmp_path: Path):
        event = tmp_path / "event.json"
        event.write_text("{not json")
        assert WorkflowContext.from_env({"GITHUB_EVENT_PATH": str(event)}).pr_number is None

    def test_missing_event_file_is_tolerated(self):
        context = WorkflowContext.from_env({"GITHUB_EVENT_PATH": "/nonexistent/event.json"})
        assert context.pr_number is None

    def test_summary_url_points_at_the_run(self):
        context = WorkflowContext.from_env(
            {"GITHUB_REPOSITORY": "o/r", "GITHUB_RUN_ID": "9"}
        )
        assert context.summary_url == "https://github.com/o/r/actions/runs/9"

    def test_summary_url_is_none_outside_actions(self):
        assert WorkflowContext.from_env({}).summary_url is None


class TestStepSummary:
    def test_writes_to_the_summary_file(self, tmp_path: Path):
        summary = tmp_path / "summary.md"
        context = WorkflowContext.from_env({"GITHUB_STEP_SUMMARY": str(summary)})
        assert write_step_summary(context, "# hello") is True
        assert "# hello" in summary.read_text()

    def test_appends_rather_than_overwrites(self, tmp_path: Path):
        summary = tmp_path / "summary.md"
        summary.write_text("earlier step\n")
        context = WorkflowContext.from_env({"GITHUB_STEP_SUMMARY": str(summary)})
        write_step_summary(context, "later step")
        content = summary.read_text()
        assert "earlier step" in content and "later step" in content

    def test_reports_false_when_not_running_in_actions(self):
        assert write_step_summary(WorkflowContext.from_env({}), "x") is False


class TestClientValidation:
    def test_requires_a_token(self):
        with pytest.raises(GitHubError, match="GITHUB_TOKEN is required"):
            GitHubClient(token="", repo="o/r")

    def test_requires_a_qualified_repository(self):
        with pytest.raises(GitHubError, match="owner/name"):
            GitHubClient(token="t", repo="justname")


class TestStickyComment:
    def client(self, server, **kwargs):
        return GitHubClient(
            token="t", repo="o/r", base_url=server.base_url, sleep=lambda _: None, **kwargs
        )

    def test_creates_a_comment_when_none_exists(self):
        handler = github_handler()
        with FakeServer(handler) as server:
            cid, created = self.client(server).upsert_comment(4, f"{COMMENT_MARKER}\nbody")
        assert created is True
        assert handler.state["comments"][0]["id"] == cid

    def test_updates_our_existing_comment_in_place(self):
        # Otherwise every push adds another comment to the thread.
        handler = github_handler(
            comments=[{"id": 11, "body": f"{COMMENT_MARKER}\nold body"}]
        )
        with FakeServer(handler) as server:
            cid, created = self.client(server).upsert_comment(4, f"{COMMENT_MARKER}\nnew body")
        assert (cid, created) == (11, False)
        assert handler.state["comments"][0]["body"].endswith("new body")
        assert len(handler.state["comments"]) == 1

    def test_ignores_comments_from_other_authors(self):
        handler = github_handler(
            comments=[
                {"id": 1, "body": "a human review comment"},
                {"id": 2, "body": "<!-- someothertool -->"},
            ]
        )
        with FakeServer(handler) as server:
            cid, created = self.client(server).upsert_comment(4, f"{COMMENT_MARKER}\nx")
        assert created is True
        assert cid not in (1, 2)

    def test_paginates_when_a_thread_has_many_comments(self):
        many = [{"id": i, "body": f"comment {i}"} for i in range(COMMENT_PAGE_SIZE)]
        pages = {"n": 0}

        def handler(method, path, body):
            if method == "GET":
                pages["n"] += 1
                # Match the page parameter exactly: 'page=1' is also a substring of
                # 'per_page=100', which would make every page look like the first.
                if "&page=1" in path:
                    return 200, many
                return 200, [{"id": 999, "body": f"{COMMENT_MARKER}\nfound on page 2"}]
            return 200, {"id": 1, "body": ""}

        with FakeServer(handler) as server:
            cid, created = self.client(server).upsert_comment(4, f"{COMMENT_MARKER}\nx")
        assert (cid, created) == (999, False)
        assert pages["n"] == 2

    def test_comment_search_is_bounded_when_every_page_is_full(self):
        # A thread that always returns a full page must not spin the job forever.
        full = [{"id": i, "body": "unrelated"} for i in range(COMMENT_PAGE_SIZE)]
        pages = {"n": 0}

        def handler(method, path, body):
            if method == "GET":
                pages["n"] += 1
                return 200, full
            return 201, {"id": 7, "body": body["body"]}

        with FakeServer(handler) as server:
            cid, created = self.client(server).upsert_comment(4, f"{COMMENT_MARKER}\nx")
        assert pages["n"] == MAX_COMMENT_PAGES
        assert created is True  # falls back to posting a fresh comment

    def test_refuses_a_body_over_the_api_limit(self):
        # The renderer truncates; this is the backstop if it ever fails to.
        with FakeServer(github_handler()) as server:
            with pytest.raises(GitHubError, match="over GitHub's"):
                self.client(server).upsert_comment(4, "x" * 70_000)

    def test_retries_a_transient_failure(self):
        handler = github_handler(failures=[502])
        with FakeServer(handler) as server:
            _, created = self.client(server).upsert_comment(4, f"{COMMENT_MARKER}\nx")
        assert created is True

    def test_does_not_retry_a_permission_error(self):
        handler = github_handler(failures=[403])
        with FakeServer(handler) as server:
            with pytest.raises(GitHubError, match="HTTP 403"):
                self.client(server).upsert_comment(4, "x")

    def test_sends_authorization_and_api_version_headers(self):
        with FakeServer(github_handler()) as server:
            self.client(server).upsert_comment(4, f"{COMMENT_MARKER}\nx")
            headers = {k.lower(): v for k, v in server.requests[0][3].items()}
        assert headers["authorization"] == "Bearer t"
        assert headers["x-github-api-version"] == "2022-11-28"


class TestCheckRun:
    def test_posts_a_completed_check_with_a_conclusion(self):
        handler = github_handler()
        with FakeServer(handler) as server:
            client = GitHubClient(token="t", repo="o/r", base_url=server.base_url)
            client.create_check_run(
                name="Token Report",
                head_sha="abc",
                conclusion="failure",
                title="over budget",
                summary="details",
            )
        run = handler.state["checks"][0]
        assert run["status"] == "completed"
        assert run["conclusion"] == "failure"
        assert run["output"]["title"] == "over budget"

    def test_caps_annotations_at_the_api_limit(self):
        handler = github_handler()
        annotations = [
            {
                "path": f"p{i}.md",
                "start_line": 1,
                "end_line": 1,
                "annotation_level": "failure",
                "message": "m",
            }
            for i in range(MAX_ANNOTATIONS + 20)
        ]
        with FakeServer(handler) as server:
            client = GitHubClient(token="t", repo="o/r", base_url=server.base_url)
            client.create_check_run(
                name="n", head_sha="s", conclusion="failure", title="t",
                summary="s", annotations=annotations,
            )
        assert len(handler.state["checks"][0]["output"]["annotations"]) == MAX_ANNOTATIONS


class TestAnnotations:
    def budgets(self, **kwargs):
        head = report({"big": 900, "small": 10})
        comparison = compare(head, None)
        return comparison, evaluate(comparison, Budgets(**kwargs))

    def test_component_violations_become_annotations(self):
        _, budgets = self.budgets(component={"big": 100})
        annotations = budget_annotations(budgets, changed_paths=["prompts/big.md"])
        assert len(annotations) == 1
        assert annotations[0]["path"] == "prompts/big.md"
        assert annotations[0]["annotation_level"] == "failure"

    def test_violations_outside_the_diff_are_skipped(self):
        # GitHub discards annotations for untouched files, so filter them out.
        _, budgets = self.budgets(component={"big": 100})
        assert budget_annotations(budgets, changed_paths=["unrelated.py"]) == []

    def test_all_violations_annotate_when_the_diff_is_unknown(self):
        _, budgets = self.budgets(component={"big": 100})
        assert len(budget_annotations(budgets, changed_paths=None)) == 1

    def test_total_budget_violations_do_not_annotate_a_file(self):
        # A repo-wide ceiling belongs to no single line.
        _, budgets = self.budgets(resident_total=10)
        assert budget_annotations(budgets, changed_paths=None) == []


class TestCheckSummary:
    def test_conclusion_reflects_budget_state(self):
        head = report({"a": 5000})
        passing = evaluate(compare(head, None), Budgets(resident_total=9000))
        failing = evaluate(compare(head, None), Budgets(resident_total=10))
        assert check_conclusion(passing) == "success"
        assert check_conclusion(failing) == "failure"

    def test_title_leads_with_violations_when_over_budget(self):
        head = report({"a": 5000})
        comparison = compare(head, None)
        budgets = evaluate(comparison, Budgets(resident_total=10))
        assert check_title(comparison, budgets).startswith("1 budget violation")

    def test_title_shows_the_delta_when_within_budget(self):
        comparison = compare(report({"a": 130}), report({"a": 100}))
        title = check_title(comparison, evaluate(comparison, Budgets()))
        assert "+30" in title

    def test_title_notes_an_unchanged_total(self):
        comparison = compare(report({"a": 100}), report({"a": 100}))
        assert "unchanged" in check_title(comparison, evaluate(comparison, Budgets()))

    def test_title_omits_a_delta_without_a_baseline(self):
        comparison = compare(report({"a": 100}), None)
        title = check_title(comparison, evaluate(comparison, Budgets()))
        assert "vs base" not in title
