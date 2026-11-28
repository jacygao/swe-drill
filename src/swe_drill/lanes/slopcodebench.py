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
import shlex
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
# A pin of the scorer, as one word of an install command, with its version where it gives one.
_PIN = re.compile(r"scb-check(?:==([0-9A-Za-z.]+))?")
# Shell commands that install Python packages, by their first words.
_INSTALLERS = (
    ("uv", "tool", "install"),
    ("uv", "pip", "install"),
    ("pip", "install"),
    ("pip3", "install"),
    ("python", "-m", "pip", "install"),
    ("python3", "-m", "pip", "install"),
)
# Where one shell command ends and the next begins.
_OPERATORS = {"&&", "||", ";", "|", "&"}
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_ESCAPED_OPERATOR = re.compile(r"\\[;&|]")


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
    try:
        installs = _scorer_installs(dockerfile)
    except ValueError as error:
        raise LaneError(
            f"{task / DOCKERFILE}: a RUN instruction cannot be read as shell ({error}), so the lane"
            " cannot tell whether it installs scb-check"
        ) from None
    if installs != [SCB_CHECK]:
        raise LaneError(
            f"{task / DOCKERFILE}: installs scb-check as {', '.join(installs) or 'nothing'},"
            f" but the lane records scores from scb-check {SCB_CHECK}, installed once"
        )


def _scorer_installs(dockerfile: bytes) -> list[str]:
    """
    The version of each install of scb-check by the Dockerfile's RUN
    instructions, or "unpinned" where one pins none.

    Each instruction that names scb-check is split into words and commands,
    quotes and comments included, so a pin anywhere but among an install
    command's own words, such as in an echo's argument, a comment or a LABEL,
    installs nothing and is not counted. Raises ValueError for such an
    instruction the split cannot read, such as one whose quote is never
    closed or that escapes an operator, so that a form it cannot establish is
    refused rather than counted.
    """
    versions = []
    for instruction in _run_instructions(dockerfile):
        text = instruction.decode(errors="replace")
        # One that never names the scorer cannot install it, and need not be read.
        if "scb-check" not in text:
            continue
        # Read without escapes below, so an escaped operator would pass for one the shell runs.
        if _ESCAPED_OPERATOR.search(text):
            raise ValueError("a backslash before ;, & or |")
        # Not POSIX, so a quoted operator keeps its quotes and is never taken for one.
        lexer = shlex.shlex(text, posix=False, punctuation_chars=True)
        lexer.whitespace_split = True
        command: list[str] = []
        for word in [*lexer, ";"]:
            if word not in _OPERATORS:
                command.append(_unquoted(word))
                continue
            versions += _pins(command)
            command = []
    return versions


def _unquoted(word: str) -> str:
    """The word as the shell passes it on, its quotes removed."""
    parts = shlex.split(word)
    return parts[0] if parts else ""


def _pins(command: list[str]) -> list[str]:
    """The scb-check versions one shell command installs, after any VAR=value it sets."""
    while command and _ASSIGNMENT.match(command[0]):
        command = command[1:]
    for installer in _INSTALLERS:
        if tuple(command[: len(installer)]) == installer:
            pins = (_PIN.fullmatch(word) for word in command[len(installer) :])
            return [pin.group(1) or "unpinned" for pin in pins if pin]
    return []


def _run_instructions(dockerfile: bytes) -> list[bytes]:
    """Each RUN instruction's shell text, its continued lines joined, with no comment line in it."""
    instructions, current = [], b""
    for line in dockerfile.splitlines():
        if line.lstrip().startswith(b"#"):
            continue
        stripped = line.rstrip()
        if stripped.endswith(b"\\"):
            current += stripped[:-1] + b" "
            continue
        current += line
        instruction, current = current.strip(), b""
        if instruction[:4].upper() == b"RUN ":
            instructions.append(instruction[4:])
    return instructions
