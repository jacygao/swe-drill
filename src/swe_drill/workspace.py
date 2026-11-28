"""
Find the workspace, and capture everything a harness changed in it as one patch.

Every command runs in the sandbox, through Harbor's environment, since the
workspace is there and not on the machine running SWE-drill.
"""

import shlex
from collections.abc import Sequence

from harbor.environments.base import BaseEnvironment

LOGS_DIR = "/logs/agent"
# What Harbor collects from the agent's sandbox for a verifier in a sandbox of its own.
ARTIFACTS_DIR = "/logs/artifacts"
PATCH_NAME = "model.patch"
# A later step's own change, which model.patch holds with every step before it.
STEP_PATCH_NAME = "step.patch"


# A commit from SWE-drill's own identity, which no config, hook or init
# template of the image signs, rejects or changes.
_COMMIT = (
    "-c user.name=SWE-drill -c user.email=swe-drill@localhost"
    " -c commit.gpgsign=false -c core.hooksPath=/dev/null commit -q"
)


class WorkspaceError(Exception):
    """A workspace SWE-drill cannot read or capture. The message carries git's output."""


async def find_workspace(env: BaseEnvironment) -> str:
    """
    Return the directory the sandbox starts in, which is the image's WORKDIR.

    It must be the top level of a git repository. A directory inside one is
    refused as well, because `git apply` there skips every path outside it,
    so the verifier would grade part of the patch. Raises WorkspaceError naming
    the directory when it is not.
    """
    # Only the newline pwd ends with, since a directory's name can end in whitespace.
    start = (await _run(env, "pwd")).removesuffix("\n")
    # Into stdout, because Modal keeps stderr apart and git explains a failure there.
    result = await env.exec(f"git -C {shlex.quote(start)} rev-parse --show-prefix 2>&1")
    if result.return_code != 0:
        raise WorkspaceError(
            f"the sandbox starts in {start}, where git finds no repository it can read:"
            f" {(result.stdout or '').strip()}"
        )
    prefix = (result.stdout or "").strip()
    if prefix:
        raise WorkspaceError(
            f"the sandbox starts in {start}, which is {prefix} inside a git repository,"
            " not its top level"
        )
    return start


async def make_workspace(env: BaseEnvironment) -> str:
    """
    Return the workspace as `find_workspace` does. An empty start directory is
    first made a repository at an empty commit.

    A task whose agent writes its code from nothing starts in an empty
    directory. One that holds files and no repository is refused as
    `find_workspace` refuses it, since the image has then lost the repository
    its files belong to. The repository is made by the sandbox's default user,
    who runs the agent's commands too, and marked safe in that user's git
    config whoever owns the directory.
    """
    start = (await _run(env, "pwd")).removesuffix("\n")
    quoted = shlex.quote(start)
    # Exits 0 only when git finds no repository and the directory holds nothing.
    empty = await env.exec(
        f"! git -C {quoted} rev-parse --git-dir >/dev/null 2>&1"
        f' && [ -z "$(ls -A {quoted})" ]'
    )
    if empty.return_code == 0:
        # Git refuses a repository in a directory another user owns, as an
        # image's WORKDIR often is, to this user and to the agent alike.
        await _run(
            env,
            f"git config --global --add safe.directory {quoted} 2>&1"
            f" && git -C {quoted} init -q 2>&1"
            f" && git -C {quoted} {_COMMIT} --allow-empty -m 'Empty workspace' 2>&1",
        )
    return await find_workspace(env)


async def commit_workspace(env: BaseEnvironment, workspace: str) -> None:
    """
    Commit everything the workspace holds that git does not ignore, so the
    next step of a task starts on a commit of the step before.

    Raises WorkspaceError when git cannot.
    """
    git = f"git -C {shlex.quote(workspace)}"
    # Empty when the step before changed nothing, so every step still starts on a commit of its own.
    await _run(env, f"{git} add -A 2>&1 && {git} {_COMMIT} --allow-empty -m 'Start of step' 2>&1")


async def read_base(env: BaseEnvironment, workspace: str) -> str:
    """
    Return the commit the workspace's HEAD names.

    Read it before the harness runs, since a harness that commits moves HEAD.
    Raises WorkspaceError when git cannot read it.
    """
    output = await _run(env, f"git -C {shlex.quote(workspace)} rev-parse --verify HEAD")
    return output.strip()


async def read_start_tree(env: BaseEnvironment, workspace: str) -> str:
    """
    Return the tree the workspace holds: HEAD's, with every change the working
    tree has, staged or not, and every untracked file git does not ignore.

    Read it before the harness runs. It is built in a copy of the index, so the
    index, HEAD, the refs and `git status` are as they were. Raises
    WorkspaceError when git cannot build it.
    """
    # A copy of the real index, or none where there is none, so that `git add -A`
    # here keeps and drops the same files git ignores as it does at capture.
    output = await _run(
        env,
        "{"
        f" cd {shlex.quote(workspace)}"
        " && index=$(git rev-parse --git-path index)"
        " && copy=$(mktemp -d)"
        ' && export GIT_INDEX_FILE="$copy/index"'
        ' && { [ ! -f "$index" ] || cp "$index" "$GIT_INDEX_FILE"; }'
        " && git add -A && git write-tree; } 2>&1;"
        ' status=$?; [ -n "$copy" ] && rm -rf "$copy"; exit $status',
    )
    # The tree is write-tree's line, the last; add -A can warn before it.
    return output.split()[-1]


async def capture_patch(
    env: BaseEnvironment,
    workspace: str,
    start: str,
    logs_dir: str = LOGS_DIR,
    name: str = PATCH_NAME,
    leave_out: Sequence[str] = (),
) -> int:
    """
    Write the whole change from the tree `start` names to `name` in the logs
    directory, and return its size in bytes, leaving out the workspace paths
    in `leave_out` and everything under them.

    The change is the same whether the harness committed it, staged it or left
    it in the working tree. A workspace with no change gives an empty patch.
    Raises WorkspaceError when git fails, so an empty patch only ever means no
    change.
    """
    git = f"git -C {shlex.quote(workspace)}"
    patch = shlex.quote(f"{logs_dir.rstrip('/')}/{name}")
    await _run(env, f"mkdir -p {shlex.quote(logs_dir)} && {git} add -A")
    # A file in the starting tree that git now ignores is still staged while it
    # exists, as a tracked file would be. Otherwise a harness that only ignores
    # a file the image left untracked would give a patch deleting it.
    # Only the starting files add -A left out can be such a file. Each is
    # checked alone, because check-ignore exits 128 for a path beyond a symlink,
    # which a directory the harness replaced with one leaves, and that path is
    # a real deletion.
    await _run(
        env,
        "{"
        f" cd {shlex.quote(workspace)}"
        " && list=$(mktemp)"
        f' && git diff --cached -z --no-renames --name-only --diff-filter=D {shlex.quote(start)} > "$list"'
        " && while IFS= read -r -d '' path; do"
        ' if { [ -L "$path" ] || [ -f "$path" ]; }'
        ' && git check-ignore -q --no-index -- "$path" 2>/dev/null; then printf "%s\\0" "$path"; fi;'
        ' done < "$list" | git update-index -z --add --stdin; } 2>&1;'
        ' status=$?; [ -n "$list" ] && rm -f "$list"; exit $status',
    )
    excluded = " ".join(shlex.quote(f":(exclude,literal,top){path}") for path in leave_out)
    # Git writes the file itself, because exec returns stdout as decoded text,
    # which corrupts any byte that is not valid UTF-8. The flags after --binary
    # stop the repository's config writing a patch that does not apply.
    await _run(
        env,
        f"{git} diff --cached --binary"
        " --no-color --no-ext-diff --no-textconv --src-prefix=a/ --dst-prefix=b/"
        f" {shlex.quote(start)} -- . {excluded} > {patch}",
    )
    return int(await _run(env, f"wc -c < {patch}"))


async def _run(env: BaseEnvironment, command: str) -> str:
    result = await env.exec(command)
    if result.return_code != 0:
        raise WorkspaceError(
            f"`{command}` exited {result.return_code}: {(result.stdout or '').strip()}"
        )
    return result.stdout or ""
