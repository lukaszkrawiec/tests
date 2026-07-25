"""Real git, real repositories.

The token arithmetic is easy to get right; the git plumbing is where data quietly goes
missing. These tests run against actual repositories in temp directories.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tokenreport.gitio import Git, GitError, push_files, resolve_baseline_ref
from tokenreport.history import History, HistoryEntry


def run(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)}: {result.stderr}")
    return result.stdout.strip()


@pytest.fixture
def remote(tmp_path: Path) -> Path:
    path = tmp_path / "remote.git"
    path.mkdir()
    run("init", "--bare", "--initial-branch=main", ".", cwd=path)
    return path


@pytest.fixture
def clone(tmp_path: Path, remote: Path):
    counter = {"n": 0}

    def make() -> Path:
        counter["n"] += 1
        path = tmp_path / f"clone{counter['n']}"
        run("clone", str(remote), str(path), cwd=tmp_path)
        run("config", "user.email", "t@example.test", cwd=path)
        run("config", "user.name", "Test", cwd=path)
        return path

    return make


@pytest.fixture
def seeded(clone):
    """A clone with one commit on main, pushed to the remote."""
    path = clone()
    (path / "README.md").write_text("seed\n")
    run("add", "-A", cwd=path)
    run("commit", "-m", "seed", cwd=path)
    run("push", "-u", "origin", "main", cwd=path)
    return path


def entry(commit: str, resident: int) -> HistoryEntry:
    return HistoryEntry(
        commit=commit,
        timestamp="2026-07-01T00:00:00Z",
        model="claude-opus-5",
        counter="anthropic",
        resident=resident,
        on_demand=0,
    )


def read_history(remote: Path, branch: str, path: str = "history.json") -> History:
    return History.loads(run("show", f"{branch}:{path}", cwd=remote))


class TestPushFiles:
    def test_creates_the_branch_when_it_does_not_exist(self, seeded, remote):
        git = Git(root=seeded)
        result = push_files(
            git,
            branch="token-report-data",
            build=lambda parent: {"history.json": History(entries=[entry("a", 10)]).dumps()},
            message="first",
        )
        assert result.attempts == 1
        assert read_history(remote, "token-report-data").entries[0].resident == 10

    def test_the_history_branch_is_an_orphan(self, seeded, remote):
        # It must not carry the source tree, or every run would rewrite the whole repo.
        git = Git(root=seeded)
        push_files(
            git,
            branch="token-report-data",
            build=lambda parent: {"history.json": "{}\n"},
            message="first",
        )
        files = run("ls-tree", "--name-only", "token-report-data", cwd=remote).split()
        assert files == ["history.json"]

    def test_appends_a_second_commit_preserving_earlier_entries(self, seeded, remote):
        git = Git(root=seeded)

        def build_with(new):
            def build(parent):
                existing = git.show(parent, "history.json") if parent else None
                return {"history.json": History.loads(existing).append(new).dumps()}

            return build

        push_files(git, branch="data", build=build_with(entry("a", 10)), message="1")
        push_files(git, branch="data", build=build_with(entry("b", 20)), message="2")

        history = read_history(remote, "data")
        assert {e.commit for e in history.entries} == {"a", "b"}
        assert len(run("rev-list", "data", cwd=remote).splitlines()) == 2

    def test_preserves_unrelated_files_on_the_branch(self, seeded, remote):
        git = Git(root=seeded)
        push_files(
            git,
            branch="data",
            build=lambda p: {"history.json": "{}\n", "index.html": "<p>dash</p>"},
            message="1",
        )
        push_files(git, branch="data", build=lambda p: {"history.json": "[]\n"}, message="2")
        files = sorted(run("ls-tree", "--name-only", "data", cwd=remote).split())
        assert files == ["history.json", "index.html"]

    def test_supports_nested_paths(self, seeded, remote):
        git = Git(root=seeded)
        push_files(
            git,
            branch="data",
            build=lambda p: {"dev/bench/data.json": "{}\n"},
            message="1",
        )
        assert run("show", "data:dev/bench/data.json", cwd=remote) == "{}"

    def test_identical_content_does_not_add_an_empty_commit(self, seeded, remote):
        git = Git(root=seeded)
        push_files(git, branch="data", build=lambda p: {"history.json": "{}\n"}, message="1")
        before = run("rev-parse", "data", cwd=remote)
        push_files(git, branch="data", build=lambda p: {"history.json": "{}\n"}, message="2")
        assert run("rev-parse", "data", cwd=remote) == before

    def test_does_not_disturb_the_working_tree(self, seeded):
        # CI has the source branch checked out; switching branches under it loses work.
        git = Git(root=seeded)
        (seeded / "uncommitted.txt").write_text("work in progress\n")
        push_files(git, branch="data", build=lambda p: {"history.json": "{}\n"}, message="1")
        assert (seeded / "uncommitted.txt").read_text() == "work in progress\n"
        assert run("rev-parse", "--abbrev-ref", "HEAD", cwd=seeded) == "main"
        assert (seeded / "README.md").exists()


class TestConcurrentAppend:
    """Two merges landing seconds apart must not lose a data point.

    This is the failure that makes hand-rolled versions of this tool quietly wrong: both
    jobs read the same history, both append their own entry, and whichever pushes second
    either fails or overwrites the first.
    """

    def test_the_loser_of_a_race_retries_and_both_entries_survive(
        self, seeded, clone, remote
    ):
        first, second = seeded, clone()
        git_a, git_b = Git(root=first), Git(root=second)

        # Establish the branch with one entry.
        push_files(
            git_a,
            branch="data",
            build=lambda p: {"history.json": History(entries=[entry("base", 1)]).dumps()},
            message="base",
        )

        calls = {"n": 0}

        def build_for_a(parent):
            existing = git_a.show(parent, "history.json") if parent else None
            history = History.loads(existing)
            calls["n"] += 1
            if calls["n"] == 1:
                # Between A reading the tip and A pushing, B lands its own commit.
                def build_for_b(parent_b):
                    text = git_b.show(parent_b, "history.json") if parent_b else None
                    return {
                        "history.json": History.loads(text).append(entry("B", 20)).dumps()
                    }

                push_files(git_b, branch="data", build=build_for_b, message="from B")
            return {"history.json": history.append(entry("A", 10)).dumps()}

        result = push_files(git_a, branch="data", build=build_for_a, message="from A")

        assert result.attempts == 2, "A should have been rejected once and retried"
        history = read_history(remote, "data")
        assert {e.commit for e in history.entries} == {"base", "A", "B"}

    def test_gives_up_with_a_clear_error_after_repeated_rejection(self, seeded, clone):
        first, second = seeded, clone()
        git_a, git_b = Git(root=first), Git(root=second)
        push_files(
            git_a, branch="data", build=lambda p: {"history.json": "{}\n"}, message="base"
        )

        counter = {"n": 0}

        def always_contended(parent):
            # Someone else lands a commit before every one of A's attempts.
            counter["n"] += 1
            push_files(
                git_b,
                branch="data",
                build=lambda p: {"history.json": json.dumps({"n": counter["n"]}) + "\n"},
                message=f"contender {counter['n']}",
            )
            return {"history.json": f"// attempt {counter['n']}\n"}

        with pytest.raises(GitError, match="after 3 attempts"):
            push_files(
                git_a,
                branch="data",
                build=always_contended,
                message="from A",
                max_attempts=3,
            )


class TestResolveBaselineRef:
    def test_finds_the_merge_base_of_a_branch_and_main(self, seeded):
        git = Git(root=seeded)
        fork_point = run("rev-parse", "HEAD", cwd=seeded)
        run("checkout", "-b", "feature", cwd=seeded)
        (seeded / "new.txt").write_text("x\n")
        run("add", "-A", cwd=seeded)
        run("commit", "-m", "feature work", cwd=seeded)

        assert resolve_baseline_ref(git, base_ref="main") == fork_point

    def test_ignores_later_commits_on_the_base_branch(self, seeded, clone):
        # The baseline is the fork point, not the current tip of main: otherwise an
        # unrelated merge to main would show up as this PR's growth.
        git = Git(root=seeded)
        fork_point = run("rev-parse", "HEAD", cwd=seeded)
        run("checkout", "-b", "feature", cwd=seeded)
        (seeded / "f.txt").write_text("f\n")
        run("add", "-A", cwd=seeded)
        run("commit", "-m", "feature", cwd=seeded)

        other = clone()
        (other / "unrelated.txt").write_text("u\n")
        run("add", "-A", cwd=other)
        run("commit", "-m", "unrelated", cwd=other)
        run("push", "origin", "main", cwd=other)
        run("fetch", "origin", "main", cwd=seeded)

        assert resolve_baseline_ref(git, base_ref="main") == fork_point

    def test_returns_none_without_a_base_ref(self, seeded):
        assert resolve_baseline_ref(Git(root=seeded), base_ref=None) is None

    def test_returns_none_for_an_unknown_base_ref(self, seeded):
        assert resolve_baseline_ref(Git(root=seeded), base_ref="no-such-branch") is None


class TestGitPrimitives:
    def test_show_returns_none_for_a_missing_path(self, seeded):
        assert Git(root=seeded).show("HEAD", "absent.json") is None

    def test_show_returns_none_for_a_missing_ref(self, seeded):
        assert Git(root=seeded).show("no-such-ref", "README.md") is None

    def test_rev_parse_returns_none_for_a_missing_ref(self, seeded):
        assert Git(root=seeded).rev_parse("no-such-ref") is None

    def test_failed_command_raises_with_the_stderr(self, seeded):
        with pytest.raises(GitError, match="failed"):
            Git(root=seeded).run("cat-file", "-p", "0" * 40)
