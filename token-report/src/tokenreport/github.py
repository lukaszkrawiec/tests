"""GitHub REST calls and workflow environment plumbing.

Only four operations are needed — list comments, create a comment, edit a comment, and
create a check run — so this speaks HTTP directly rather than taking a dependency. That
also makes it possible to point the client at a local fake and exercise the real request
path in tests.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .compare import BudgetResult, Comparison, Severity
from .render.markdown import COMMENT_LIMIT, COMMENT_MARKER

DEFAULT_API = "https://api.github.com"
API_VERSION = "2022-11-28"
# GitHub rejects a check run carrying more than 50 annotations in one request.
MAX_ANNOTATIONS = 50
COMMENT_PAGE_SIZE = 100
# Bounds the comment search so a pathological thread cannot spin the job forever.
MAX_COMMENT_PAGES = 20


class GitHubError(RuntimeError):
    """Raised when a GitHub API call fails."""


@dataclass(frozen=True)
class WorkflowContext:
    """The parts of the Actions environment this tool needs."""

    repo: str | None = None
    sha: str | None = None
    ref: str | None = None
    base_ref: str | None = None
    head_ref: str | None = None
    pr_number: int | None = None
    run_id: str | None = None
    event_name: str | None = None
    default_branch: str | None = None
    step_summary: Path | None = None
    server_url: str = "https://github.com"

    @property
    def is_pull_request(self) -> bool:
        return self.pr_number is not None

    @property
    def summary_url(self) -> str | None:
        if not (self.repo and self.run_id):
            return None
        return f"{self.server_url}/{self.repo}/actions/runs/{self.run_id}"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "WorkflowContext":
        env = env if env is not None else os.environ
        pr_number = None
        event_path = env.get("GITHUB_EVENT_PATH")
        if event_path and Path(event_path).is_file():
            try:
                payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                payload = {}
            number = (payload.get("pull_request") or {}).get("number")
            if isinstance(number, int):
                pr_number = number
            default_branch = (payload.get("repository") or {}).get("default_branch")
        else:
            default_branch = None

        summary = env.get("GITHUB_STEP_SUMMARY")
        return cls(
            repo=env.get("GITHUB_REPOSITORY"),
            sha=env.get("GITHUB_SHA"),
            ref=env.get("GITHUB_REF"),
            base_ref=env.get("GITHUB_BASE_REF") or None,
            head_ref=env.get("GITHUB_HEAD_REF") or None,
            pr_number=pr_number,
            run_id=env.get("GITHUB_RUN_ID"),
            event_name=env.get("GITHUB_EVENT_NAME"),
            default_branch=default_branch,
            step_summary=Path(summary) if summary else None,
            server_url=env.get("GITHUB_SERVER_URL", "https://github.com"),
        )


def write_step_summary(context: WorkflowContext, markdown: str) -> bool:
    """Append to the job summary. Returns False when not running in Actions."""
    if context.step_summary is None:
        return False
    with context.step_summary.open("a", encoding="utf-8") as handle:
        handle.write(markdown)
        handle.write("\n")
    return True


class GitHubClient:
    def __init__(
        self,
        *,
        token: str,
        repo: str,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        sleep=time.sleep,
    ) -> None:
        if not token:
            raise GitHubError("a GITHUB_TOKEN is required to post to the API")
        if not repo or "/" not in repo:
            raise GitHubError(f"repository must look like 'owner/name', got {repo!r}")
        self._token = token
        self._repo = repo
        self._base = (base_url or os.environ.get("GITHUB_API_URL") or DEFAULT_API).rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = sleep

    def _request(self, method: str, path: str, payload: Any = None) -> Any:
        url = path if path.startswith("http") else f"{self._base}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "accept": "application/vnd.github+json",
                "authorization": f"Bearer {self._token}",
                "x-github-api-version": API_VERSION,
                "content-type": "application/json",
                "user-agent": "tokenreport",
            },
        )
        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    body = response.read().decode("utf-8")
                return json.loads(body) if body.strip() else None
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", "replace")[:500]
                finally:
                    exc.close()
                if exc.code not in (429, 500, 502, 503, 504):
                    raise GitHubError(
                        f"{method} {path} returned HTTP {exc.code}: {detail}"
                    ) from None
                last_error = f"HTTP {exc.code}: {detail}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self._max_retries:
                self._sleep(2**attempt)
        raise GitHubError(f"{method} {path} failed after retries: {last_error}")

    def find_comment(self, pr_number: int, marker: str = COMMENT_MARKER) -> int | None:
        """Find our own previous comment by its hidden marker.

        Without this every push would add another comment; with it the thread keeps one
        comment showing live state.

        Paging is capped: a job that hangs forever is a worse failure than posting a
        second comment on a thread with thousands of them.
        """
        for page in range(1, MAX_COMMENT_PAGES + 1):
            comments = self._request(
                "GET",
                f"/repos/{self._repo}/issues/{pr_number}/comments"
                f"?per_page={COMMENT_PAGE_SIZE}&page={page}",
            )
            if not comments:
                return None
            for comment in comments:
                if marker in (comment.get("body") or ""):
                    return int(comment["id"])
            if len(comments) < COMMENT_PAGE_SIZE:
                return None
        return None

    def upsert_comment(self, pr_number: int, body: str) -> tuple[int, bool]:
        """Create or update the sticky comment. Returns (id, created)."""
        if len(body) > COMMENT_LIMIT:
            raise GitHubError(
                f"comment body is {len(body):,} characters, over GitHub's "
                f"{COMMENT_LIMIT:,} limit; the renderer should have truncated it"
            )
        existing = self.find_comment(pr_number)
        if existing is not None:
            self._request(
                "PATCH",
                f"/repos/{self._repo}/issues/comments/{existing}",
                {"body": body},
            )
            return existing, False
        created = self._request(
            "POST", f"/repos/{self._repo}/issues/{pr_number}/comments", {"body": body}
        )
        return int(created["id"]), True

    def create_check_run(
        self,
        *,
        name: str,
        head_sha: str,
        conclusion: str,
        title: str,
        summary: str,
        annotations: Sequence[Mapping[str, Any]] = (),
    ) -> int:
        payload: dict[str, Any] = {
            "name": name,
            "head_sha": head_sha,
            "status": "completed",
            "conclusion": conclusion,
            "output": {
                "title": title,
                "summary": summary,
                "annotations": list(annotations[:MAX_ANNOTATIONS]),
            },
        }
        result = self._request("POST", f"/repos/{self._repo}/check-runs", payload)
        return int(result["id"])


def budget_annotations(
    budgets: BudgetResult, *, changed_paths: Sequence[str] | None = None
) -> list[dict[str, Any]]:
    """Turn component budget violations into inline diff annotations.

    Only components whose source file the pull request actually touched are annotated:
    GitHub silently discards annotations for files outside the diff, and annotating an
    untouched file would be noise anyway.
    """
    touched = set(changed_paths) if changed_paths is not None else None
    annotations: list[dict[str, Any]] = []
    for violation in budgets.violations:
        if not violation.source:
            continue
        if touched is not None and violation.source not in touched:
            continue
        annotations.append(
            {
                "path": violation.source,
                "start_line": 1,
                "end_line": 1,
                "annotation_level": "failure",
                "title": f"Token budget exceeded: {violation.component_id}",
                "message": violation.message,
            }
        )
    return annotations


def check_conclusion(budgets: BudgetResult) -> str:
    return "failure" if budgets.severity is Severity.FAIL else "success"


def check_title(comparison: Comparison, budgets: BudgetResult) -> str:
    resident = comparison.head.resident
    if budgets.violations:
        count = len(budgets.violations)
        return f"{count} budget violation{'s' if count != 1 else ''} · {resident:,} resident tokens"
    if not comparison.has_baseline:
        return f"{resident:,} resident tokens"
    delta = comparison.resident_delta
    if delta == 0:
        return f"{resident:,} resident tokens · unchanged"
    return f"{resident:,} resident tokens · {delta:+,} vs base"
