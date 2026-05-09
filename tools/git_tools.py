import subprocess
import os
from pathlib import Path


class GitTools:
    """Git operations sandboxed to the workspace root."""

    def __init__(self, workspace_root: str):
        self.workspace = os.path.abspath(workspace_root)

    def _is_git_repo(self) -> bool:
        return (Path(self.workspace) / ".git").is_dir()

    def _run(self, *args: str, timeout: int = 15) -> str:
        if not self._is_git_repo():
            return "Skipped: Not a git repository."
        try:
            result = subprocess.run(
                ["git"] + list(args),
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout.strip() if result.stdout else ""
            error = result.stderr.strip() if result.stderr else ""
            if result.returncode == 0:
                return output if output else "(no output)"
            return f"Git Error: {error}"
        except FileNotFoundError:
            return "Error: git is not installed or not in PATH."
        except subprocess.TimeoutExpired:
            return "Error: git command timed out."
        except Exception as e:
            return f"Error: {str(e)}"

    # --- Public Tools ---

    def git_status(self) -> str:
        """Returns short git status."""
        return self._run("status", "--short")

    def git_diff(self, staged: bool = False) -> str:
        """Returns current diff, optionally staged only."""
        if staged:
            return self._run("diff", "--staged")
        return self._run("diff")

    def git_log(self, n: int = 10) -> str:
        """Returns last N commits in oneline format."""
        return self._run("log", f"-{n}", "--oneline")

    def git_commit(self, message: str) -> str:
        """Stages all changes and commits."""
        stage_result = self._run("add", "-A")
        if stage_result.startswith("Error") or stage_result.startswith("Skipped"):
            return stage_result
        return self._run("commit", "-m", message)

    def git_checkout(self, target: str) -> str:
        """Checkout a branch or file. Restricted to safe targets."""
        # Block dangerous patterns
        if ".." in target or target.startswith("/"):
            return "Error: Unsafe checkout target."
        return self._run("checkout", target)

    def git_stash(self) -> str:
        """Stash current changes (checkpoint)."""
        return self._run("stash", "push", "-m", "999-checkpoint")

    def git_stash_pop(self) -> str:
        """Restore last stashed changes (undo)."""
        return self._run("stash", "pop")
