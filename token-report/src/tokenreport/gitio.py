"""Git plumbing.

Writes to the history branch go through git's object database rather than a checkout:
CI already has the source branch checked out, and switching branches underneath a running
job is a good way to lose work. Building a tree, committing it, and pushing the resulting
sha touches no working tree at all.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


class GitError(RuntimeError):
    """Raised when a git command fails."""


@dataclass
class Git:
    root: Path
    remote: str = "origin"
    timeout: float = 120.0

    def run(
        self,
        *args: str,
        check: bool = True,
        stdin: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> str:
        merged = {**os.environ, **(env or {})}
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.root,
                capture_output=True,
                text=True,
                input=stdin,
                timeout=self.timeout,
                env=merged,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GitError(f"git {' '.join(args)} could not run: {exc}") from exc
        if check and result.returncode != 0:
            raise GitError(
                f"git {' '.join(args)} failed ({result.returncode}): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        return result.stdout

    def rev_parse(self, ref: str) -> str | None:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        sha = result.stdout.strip()
        return sha or None

    def fetch(self, refspec: str, *, depth: int | None = None) -> bool:
        """Fetch a refspec, returning False when the remote ref does not exist."""
        args = ["fetch", "--quiet"]
        if depth:
            args += [f"--depth={depth}"]
        args += [self.remote, refspec]
        result = subprocess.run(
            ["git", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return result.returncode == 0

    def merge_base(self, a: str, b: str) -> str | None:
        result = subprocess.run(
            ["git", "merge-base", a, b],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return result.stdout.strip() or None

    def show(self, ref: str, path: str) -> str | None:
        """Read a file at a ref, or None when either the ref or the path is absent."""
        result = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        if result.returncode != 0:
            return None
        return result.stdout

    def hash_object(self, content: str) -> str:
        return self.run("hash-object", "-w", "--stdin", stdin=content).strip()

    def commit_tree(
        self, tree: str, *, message: str, parent: str | None, author: tuple[str, str]
    ) -> str:
        args = ["commit-tree", tree, "-m", message]
        if parent:
            args += ["-p", parent]
        name, email = author
        env = {
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
        }
        return self.run(*args, env=env).strip()

    def build_tree(self, files: Mapping[str, str], *, parent: str | None) -> str:
        """Build a tree containing ``files``, preserving anything else at ``parent``.

        Uses a throwaway index so the repository's real index is untouched.
        """
        with tempfile.TemporaryDirectory() as tmp:
            index = str(Path(tmp) / "index")
            env = {"GIT_INDEX_FILE": index}
            if parent:
                self.run("read-tree", parent, env=env)
            for path, content in files.items():
                blob = self.hash_object(content)
                self.run(
                    "update-index",
                    "--add",
                    "--cacheinfo",
                    f"100644,{blob},{path}",
                    env=env,
                )
            return self.run("write-tree", env=env).strip()


@dataclass
class PushResult:
    committed: str
    attempts: int


def push_files(
    git: Git,
    *,
    branch: str,
    build: Callable[[str | None], Mapping[str, str]],
    message: str,
    author: tuple[str, str] = ("token-report", "token-report@users.noreply.github.com"),
    max_attempts: int = 5,
) -> PushResult:
    """Append a commit to ``branch``, re-deriving content if the remote moved.

    ``build`` receives the current remote tip (None when the branch does not exist yet)
    and returns the files to write. It is called again on every retry, which is the
    point: two merges landing seconds apart would otherwise each read the same history,
    append their own entry, and silently drop one of them. Re-deriving from the new tip
    means the loser of the race adds its entry on top of the winner's.
    """
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        exists = git.fetch(f"{branch}:refs/remotes/{git.remote}/{branch}")
        parent = git.rev_parse(f"refs/remotes/{git.remote}/{branch}") if exists else None

        files = build(parent)
        tree = git.build_tree(files, parent=parent)

        if parent is not None and tree == git.rev_parse(f"{parent}^{{tree}}"):
            # Nothing changed; pushing an empty commit would add noise to the branch.
            return PushResult(committed=parent, attempts=attempt)

        commit = git.commit_tree(tree, message=message, parent=parent, author=author)
        result = subprocess.run(
            ["git", "push", git.remote, f"{commit}:refs/heads/{branch}"],
            cwd=git.root,
            capture_output=True,
            text=True,
            timeout=git.timeout,
        )
        if result.returncode == 0:
            return PushResult(committed=commit, attempts=attempt)
        last_error = (result.stderr or result.stdout).strip()

    raise GitError(
        f"could not update {branch} after {max_attempts} attempts: {last_error}"
    )


def resolve_baseline_ref(git: Git, *, base_ref: str | None, head: str = "HEAD") -> str | None:
    """Find the merge-base of the current head and the pull request's base branch.

    Falls back to the base ref itself when no merge-base is available, which happens
    with a shallow clone that does not contain the fork point.
    """
    if not base_ref:
        return None
    for candidate in (
        f"refs/remotes/{git.remote}/{base_ref}",
        base_ref,
    ):
        if git.rev_parse(candidate) is None:
            continue
        base = git.merge_base(candidate, head)
        return base or candidate
    if git.fetch(f"{base_ref}:refs/remotes/{git.remote}/{base_ref}"):
        candidate = f"refs/remotes/{git.remote}/{base_ref}"
        return git.merge_base(candidate, head) or candidate
    return None


def changed_paths(git: Git, *, base: str, head: str = "HEAD") -> Sequence[str]:
    output = git.run("diff", "--name-only", f"{base}..{head}", check=False)
    return [line for line in output.splitlines() if line.strip()]
