"""
What the slopcodebench lane changes in each task's copy beyond what every lane
does.

Each copy's agent phase reaches PyPI as well as the agent's own hosts, since
most of the problems' solutions install packages. Every step is graded by its
own test.sh, in a separate verifier, on the step's /app, which is what the
task-level `[verifier] environment_mode` every lane sets gives each step that
sets no mode of its own.
"""

import re
import tomllib
from pathlib import Path

from swe_drill.lanes.edits import LaneError, read_task_file

UNGRADEABLE: tuple[str, ...] = ()

# The Harbor Hub revision the lane is built from, and what each trial's grader records.
REVISION = 4
DIGEST = "sha256:aec29354c19d3e762640ab6d7d3c63ba8fcf4895e98a5756aecb9157b5bb4ae0"
# The scorer each task's image installs, which measures verbosity and erosion.
SCB_CHECK = "0.1.3"

PYPI_HOSTS = ("pypi.org", "files.pythonhosted.org")
DOCKERFILE = "environment/Dockerfile"

# The line every lane adds under [agent], which this lane widens to PyPI.
_NO_NETWORK = b'network_mode = "no-network"'
_ALLOWLIST = b'network_mode = "allowlist"\nallowed_hosts = ' + (
    b"[" + b", ".join(b'"' + host.encode() + b'"' for host in PYPI_HOSTS) + b"]"
)
# Each install's pinned version, or empty where it pins none.
_SCB_CHECK = re.compile(rb"scb-check(?:==([0-9A-Za-z.]+))?")
# A shell command that installs Python packages, after any VAR=value it sets.
_INSTALL = re.compile(rb"^(?:\S+=\S*\s+)*(?:uv tool install|uv pip install|pip3? install|python3? -m pip install)\b")
# Where one shell command ends and the next begins.
_SEPARATOR = re.compile(rb"&&|\|\||;|\|")
# A shell comment, from a # at the start of a word to the end of the line.
_COMMENT = re.compile(rb"(?:^|\s)#.*$", re.MULTILINE)


def rewrite(task: Path, toml: bytes) -> dict[str, bytes]:
    """
    The files the task's copy writes, by path from the task: its task.toml with
    the agent's network an allowlist of PYPI_HOSTS.

    Raises LaneError naming the task when the task sets its agent's network
    itself, when a step sets a verifier mode or environment of its own, or when
    its image installs scb-check other than once, at SCB_CHECK exactly.
    """
    _check_scb_check(task)
    original = tomllib.loads(read_task_file(task, "task.toml", "build the slopcodebench lane").decode())
    agent = original.get("agent", {})
    if not isinstance(agent, dict) or "network_mode" in agent or "allowed_hosts" in agent:
        raise LaneError(
            f"{task / 'task.toml'}: sets the agent's network, which the slopcodebench lane sets itself"
        )
    for step in original.get("steps", []):
        verifier = step.get("verifier", {}) if isinstance(step, dict) else {}
        if "environment_mode" in verifier or "environment" in verifier:
            raise LaneError(
                f"{task / 'task.toml'}: step {step.get('name')!r} sets a verifier mode or environment"
                " of its own, but the lane grades every step in a separate verifier from the task's"
                " environment"
            )
    if toml.count(_NO_NETWORK) != 1:
        raise LaneError(f"{task / 'task.toml'}: the lane cannot find where it closed the agent's network")
    widened = toml.replace(_NO_NETWORK, _ALLOWLIST)
    expected = tomllib.loads(toml.decode())
    expected["agent"] = {**expected["agent"], "network_mode": "allowlist", "allowed_hosts": list(PYPI_HOSTS)}
    if tomllib.loads(widened.decode()) != expected:
        raise LaneError(f"{task / 'task.toml'}: opening PyPI to the agent changes more than its network")
    return {"task.toml": widened}


def _check_scb_check(task: Path) -> None:
    dockerfile = read_task_file(task, DOCKERFILE, "build the slopcodebench lane")
    installs = [
        found.decode() or "unpinned"
        for command in _install_commands(dockerfile)
        for found in _SCB_CHECK.findall(command)
    ]
    if installs != [SCB_CHECK]:
        raise LaneError(
            f"{task / DOCKERFILE}: installs scb-check as {', '.join(installs) or 'nothing'},"
            f" but the lane records scores from scb-check {SCB_CHECK}, installed once"
        )


def _install_commands(dockerfile: bytes) -> list[bytes]:
    """
    Each shell command of the Dockerfile's RUN instructions that installs
    Python packages, with its comments left out.

    A pin anywhere else, such as in an echo, a LABEL or a comment, installs
    nothing, so it is not counted.
    """
    installs, current = [], b""
    for line in dockerfile.splitlines():
        if line.lstrip().startswith(b"#"):
            continue
        stripped = line.rstrip()
        if stripped.endswith(b"\\"):
            current += stripped[:-1] + b" "
            continue
        current += line
        instruction, current = current.strip(), b""
        if instruction[:4].upper() != b"RUN ":
            continue
        shell = _COMMENT.sub(b"", instruction[4:])
        installs += [command.strip() for command in _SEPARATOR.split(shell) if _INSTALL.match(command.strip())]
    return installs
