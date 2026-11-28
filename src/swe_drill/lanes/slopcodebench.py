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
# Each problem of revision 4, by its task's name, and how many checkpoints it declares.
CHECKPOINTS = {
    "cfgpipe": 6,
    "circuit_eval": 8,
    "code_search": 5,
    "dag_execution": 3,
    "database_migration": 5,
    "datagate": 7,
    "dynamic_buffer": 4,
    "dynamic_config_service_api": 4,
    "env_manager": 5,
    "etl_pipeline": 5,
    "eve_industry": 6,
    "eve_jump_planner": 3,
    "eve_market_tools": 4,
    "eve_route_planner": 3,
    "execution_server": 6,
    "file_backup": 4,
    "file_merger": 4,
    "file_query_tool": 5,
    "forge": 8,
    "l2m": 5,
    "layered_config_synthesizer": 4,
    "log_query": 5,
    "meshctl": 8,
    "metric_transform_lang": 5,
    "migrate_configs": 5,
    "mocked_http": 8,
    "mvvault": 6,
    "pwd_manager": 5,
    "recli": 8,
    "rejector": 5,
    "sheeteval": 7,
    "sith": 6,
    "test_translator": 8,
    "textdrop": 6,
    "trajectory_api": 5,
    "xjq": 5,
}
# The scorer each task's image installs, which measures verbosity and erosion.
SCB_CHECK = "0.1.3"

PYPI_HOSTS = ("pypi.org", "files.pythonhosted.org")
DOCKERFILE = "environment/Dockerfile"

# The line every lane adds under [agent], which this lane widens to PyPI.
_NO_NETWORK = b'network_mode = "no-network"'
_ALLOWLIST = b'network_mode = "allowlist"\nallowed_hosts = ' + (
    b"[" + b", ".join(b'"' + host.encode() + b'"' for host in PYPI_HOSTS) + b"]"
)
# A requirement naming the scorer, as one word of an install command, spelt
# any way pip reads as the same name, with whatever follows the name.
_SCORER = re.compile(r"scb[-_.]+check(?:\[[^\]]*\])?(.*)", re.IGNORECASE)
_EXACT = re.compile(r"==([0-9A-Za-z.]+)")
# Shell commands that install Python packages, by their first words.
_INSTALLERS = (
    ("uv", "tool", "install"),
    ("uv", "pip", "install"),
    ("pip", "install"),
    ("pip3", "install"),
    ("python", "-m", "pip", "install"),
    ("python3", "-m", "pip", "install"),
)
_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")
_ESCAPED_OPERATOR = re.compile(r"\\[;&|]")
# A RUN instruction's keyword, in any case, and the whitespace Docker takes after it.
_RUN = re.compile(rb"RUN\s+", re.IGNORECASE)


def rewrite(task: Path, toml: bytes) -> dict[str, bytes]:
    """
    The files the task's copy writes, by path from the task: its task.toml with
    the agent's network an allowlist of PYPI_HOSTS.

    Raises LaneError naming the task when revision 4 has no such task, or its
    checkpoints are not that revision's, when the task sets its agent's network
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
    declared = CHECKPOINTS.get(task.name)
    steps = original.get("steps", [])
    names = [step.get("name") if isinstance(step, dict) else None for step in steps]
    if declared is None or names != [f"checkpoint_{n}" for n in range(1, declared + 1)]:
        raise LaneError(
            f"{task / 'task.toml'}: revision 4 has no task {task.name}"
            if declared is None
            else f"{task / 'task.toml'}: is not revision 4's {task.name}, which declares checkpoint_1"
            f" to checkpoint_{declared}"
        )
    for step in steps:
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
    instructions, as `_version` gives it.

    Each instruction that names scb-check must be one plain command, after
    any VAR=value it sets, read with its quotes and comments, so a pin
    anywhere but among an install command's own words, such as in an echo's
    argument, a comment or a LABEL, installs nothing and is not counted.
    Raises ValueError for such an instruction that is more than one command,
    redirects or opens a subshell, does a dry run, escapes an operator, or
    cannot be read, such as one whose quote is never closed, so that a form
    whose install cannot be established is refused rather than counted.
    """
    versions = []
    for instruction in _run_instructions(dockerfile):
        text = instruction.decode(errors="replace")
        # One that never names the scorer cannot install it, and need not be read.
        if not re.search(r"scb[-_.]+check", text, re.IGNORECASE):
            continue
        # Read without escapes below, so an escaped operator would pass for one the shell runs.
        if _ESCAPED_OPERATOR.search(text):
            raise ValueError("a backslash before ;, & or |")
        # Not POSIX, so a quoted operator keeps its quotes and is never taken for one.
        lexer = shlex.shlex(text, posix=False, punctuation_chars=True)
        lexer.whitespace_split = True
        words = list(lexer)
        # Whether a command after an operator, a redirect or a subshell runs is the shell's to decide.
        if any(word and set(word) <= set(lexer.punctuation_chars) for word in words):
            raise ValueError("the instruction that names scb-check is more than one plain command")
        versions += _pins([_unquoted(word) for word in words])
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
            if "--dry-run" in command:
                raise ValueError("a dry run installs nothing")
            return [_version(word) for word in command[len(installer) :] if _SCORER.fullmatch(word)]
    return []


def _version(requirement: str) -> str:
    """The exact version a scorer requirement pins, "unpinned" for none, or the requirement itself for any other."""
    rest = _SCORER.fullmatch(requirement).group(1)
    if not rest:
        return "unpinned"
    exact = _EXACT.fullmatch(rest)
    return exact.group(1) if exact else requirement


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
        run = _RUN.match(instruction)
        if run:
            instructions.append(instruction[run.end() :])
    return instructions
