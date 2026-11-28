"""
swe-drill leaderboard: reading every arm's trials under `results/<lane>/`, each
graded by its own verifier, and the table the command prints.

Reading never raises on what a trial left or failed to leave. One whose
verifier left no readable reward is unresolved, unless the verifier judged it
invalid.
"""

import json
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from swe_drill import config, lanes

# The keys of parsed_counts in a rebench verifier's report.
TEST_OUTCOMES = ("PASSED", "FAILED", "SKIPPED", "ERROR")


@dataclass(frozen=True)
class Trial:
    """One trial of an agent job, with what its verifier made of it."""

    # The agent job's name, `<stamp>-<i>`.
    job: str
    task: str
    patch_bytes: int
    resolved: bool
    # False when the verifier's replay.json has `"patch": null`, or when it has
    # no replay.json and raised. None when nothing says either way.
    verifier_applied: bool | None
    # True when the verifier had a patch and its replay.json does not show git
    # applied it.
    verifier_refused: bool
    # The type of the exception the verifier raised, when it wrote no replay.json.
    verifier_raised: str | None
    # Whether the verifier recorded a test result. When it records none, a 0
    # cannot tell a wrong patch from grading that broke.
    verifier_tested: bool
    # Whether the lane's judge gave a verdict on every rubric item, which a
    # lane with no judge always has. An item it gave none is left out of the
    # reward, so a judge that fails may pass a trial, or fail it, unseen.
    verifier_judged: bool
    # True when the verifier judged the trial invalid by writing an
    # empty reward, which Harbor records as RewardFileEmptyError. A timeout or
    # a crash also leaves no reward, and is not this.
    invalid: bool
    # The agent trial's directory.
    path: Path


def read_trials(directory: Path, arm: str, lane: str = lanes.REBENCH) -> list[Trial]:
    """
    Every trial in the arm's agent jobs under `results/<lane>/<arm>/`, in
    `directory`, sorted by job and task.

    An arm with no such directory has no trials.
    """
    arm_directory = Path(directory) / "results" / lane / arm
    if not arm_directory.is_dir():
        return []
    trials = []
    for job in sorted(_directories(arm_directory)):
        for trial in _directories(job):
            task = _task(trial)
            # A trial that names no task cannot be counted against one.
            if task is None:
                continue
            trials.append(_trial(job.name, task, trial, lane))
    return sorted(trials, key=lambda trial: (trial.job, trial.task))


def _trial(job: str, task: str, trial: Path, lane: str) -> Trial:
    """The trial, as its own verifier on `lane` graded it."""
    record = _json(trial / "agent" / "suss.json")
    patch_bytes = record.get("patch_bytes") if isinstance(record, dict) else None
    replay_path = trial / "verifier" / "replay.json"
    replay = _json(replay_path)
    # The verifier writes replay.json before any test, so one that raised
    # before writing it counts as applying nothing.
    raised = _exception(trial) if not replay_path.exists() else None
    if isinstance(replay, dict) and "patch" in replay:
        applied = replay["patch"] is not None
    else:
        applied = False if raised is not None else None
    return Trial(
        job=job,
        task=task,
        patch_bytes=patch_bytes if isinstance(patch_bytes, int) else 0,
        resolved=_reward(trial) == 1,
        verifier_applied=applied,
        verifier_refused=applied is True and not _applied_whole(replay),
        verifier_raised=raised,
        verifier_tested=_tested(trial) if lane == lanes.REBENCH else _tests_rewarded(trial),
        verifier_judged=lane != lanes.ATLAS_RF or _judged(trial),
        invalid=_exception(trial) == "RewardFileEmptyError",
        path=trial,
    )


def _applied_whole(replay: dict) -> bool:
    """
    Whether the replay.json of a verifier that applied a patch shows that all of
    it applied, which `git apply` does exactly when it exits 0.
    """
    code = replay.get("apply_rc")
    return type(code) is int and code == 0


def _directories(parent: Path) -> list[Path]:
    """
    The directories in parent, leaving out a job's own files.

    Harbor names each trial's directory for its task, and starts the name of
    any directory of its own with a `.`.
    """
    return [entry for entry in parent.iterdir() if entry.is_dir() and not entry.name.startswith(".")]


def _task(trial: Path) -> str | None:
    """
    The trial's task, as the last part of `task_name` in its result.json.

    A trial Harbor stopped before writing result.json still names its task in
    config.json. None when neither does.
    """
    result = _json(trial / "result.json")
    if isinstance(result, dict) and isinstance(result.get("task_name"), str):
        return result["task_name"].rsplit("/", 1)[-1]
    config = _json(trial / "config.json")
    task = config.get("task") if isinstance(config, dict) else None
    if isinstance(task, dict) and isinstance(task.get("path"), str):
        return task["path"].rstrip("/").rsplit("/", 1)[-1]
    return None


def _reward(trial: Path) -> float | int | None:
    """`verifier_result.rewards.reward` in the trial's result.json, or None."""
    result = _json(trial / "result.json")
    verifier = result.get("verifier_result") if isinstance(result, dict) else None
    rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
    reward = rewards.get("reward") if isinstance(rewards, dict) else None
    return reward if isinstance(reward, int | float) and not isinstance(reward, bool) else None


def _tested(trial: Path) -> bool:
    """
    Whether `parsed_counts` in the trial's verifier/report.json counts a test.

    A report that is missing, or whose parsed_counts is not counts, counts none.
    """
    report = _json(trial / "verifier" / "report.json")
    found = report.get("parsed_counts") if isinstance(report, dict) else None
    if not isinstance(found, dict):
        return False
    return any(
        isinstance(count, int) and not isinstance(count, bool) and count > 0
        for outcome, count in found.items()
        if outcome in TEST_OUTCOMES
    )


def _tests_rewarded(trial: Path) -> bool:
    """
    Whether the trial's verifier/comparison_results.json holds a numeric
    `tests_reward`, which SWE Atlas's test.sh writes there only when it read
    the tests' results.
    """
    found = _json(trial / "verifier" / "comparison_results.json")
    reward = found.get("tests_reward") if isinstance(found, dict) else None
    return isinstance(reward, int | float) and not isinstance(reward, bool)


def _judged(trial: Path) -> bool:
    """
    Whether the trial's verifier/rubrics_results.json gives a verdict on every
    rubric item.

    SWE Atlas's evaluate_rubrics.py marks an item `"evaluable": false` when its
    model gave no answer it could read, as when every call failed. It writes
    no `rubric_scores` when it had no key, and no file when it crashed.
    """
    found = _json(trial / "verifier" / "rubrics_results.json")
    scores = found.get("rubric_scores") if isinstance(found, dict) else None
    return isinstance(scores, list) and all(
        isinstance(score, dict) and score.get("evaluable") is True for score in scores
    )


def _exception(trial: Path) -> str | None:
    """
    The type of the exception Harbor recorded as `exception_info` in the
    trial's result.json, `an exception` when it names none, or None.
    """
    result = _json(trial / "result.json")
    info = result.get("exception_info") if isinstance(result, dict) else None
    if info is None:
        return None
    kind = info.get("exception_type") if isinstance(info, dict) else None
    return kind if isinstance(kind, str) and kind else "an exception"


def _json(path: Path):
    """The file's JSON, or None when it is missing or not JSON."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


RESAMPLES = 10_000

# Fixed, so the same results always print the same margin.
SEED = 0


def margin(outcomes: Mapping[str, Sequence[bool]]) -> float:
    """
    Half the width of a 95% interval on the resolved rate, resolved trials
    over trials, of `outcomes`, which map each task to its trials' outcomes.

    The interval comes from a bootstrap: each resample draws tasks with
    replacement and takes every trial of each task drawn, so a task's runs
    stay together.
    """
    # Per task: resolved, then trials.
    counts = [(sum(trials), len(trials)) for trials in outcomes.values()]
    rng = random.Random(SEED)
    resampled = [_rate(rng.choices(counts, k=len(counts))) for _ in range(RESAMPLES)]
    # Cut points at every 2.5%: the first is the 2.5th percentile, the last the 97.5th.
    cuts = statistics.quantiles(resampled, n=40, method="inclusive")
    return (cuts[-1] - cuts[0]) / 2


def _rate(counts: Sequence[tuple[int, int]]) -> float:
    resolved, trials = map(sum, zip(*counts))
    return resolved / trials


class LeaderboardError(Exception):
    """The results cannot be ranked, for the reason the message gives."""


@dataclass(frozen=True)
class Standing:
    """An arm's row of the leaderboard."""

    arm: str
    runs: int
    rate: float
    margin: float
    # The share of the arm's tasks that some run resolved.
    pass_at_k: float


def report(directory: Path, lane: str = lanes.REBENCH) -> str:
    """
    What `swe-drill leaderboard --lane <lane>` prints, run in `directory`, the
    directory holding swe-drill.toml, for any lane.

    Each arm is a directory under `results/<lane>/`. An invalid trial counts
    nowhere, so a task whose every trial in an arm is invalid is not one of
    that arm's tasks. The table goes by resolved rate, best first, then by arm
    name. After it come the valid trials whose verifier recorded no test
    result, then those whose judge gave no verdict on some rubric item, then
    the invalid trials, each list by arm in the table's order, then by job and
    task.

    Raises ConfigError when swe-drill.toml is missing or an arm in it sets
    `baseline`, and LeaderboardError when no arm has results, an arm has no
    trials or only invalid ones, or a trial's verifier applied none of its
    patch or could not apply it.
    """
    if lane == lanes.SLOPCODEBENCH:
        return quality_report(directory)
    config.check_arms(Path(directory) / "swe-drill.toml")
    trials = _read(directory, lane)
    standings = sorted(
        (_standing(arm, found) for arm, found in trials.items()),
        key=lambda standing: (-standing.rate, standing.arm),
    )
    order = [standing.arm for standing in standings]
    untested = [
        (arm, trial.job, trial.task)
        for arm in order
        for trial in trials[arm]
        if not trial.invalid and not trial.verifier_tested
    ]
    unjudged = [
        (arm, trial.job, trial.task)
        for arm in order
        for trial in trials[arm]
        if not trial.invalid and not trial.verifier_judged
    ]
    invalid = [(arm, trial.job, trial.task) for arm in order for trial in trials[arm] if trial.invalid]
    return render(standings, untested, invalid, unjudged)


def _standing(arm: str, trials: Sequence[Trial]) -> Standing:
    by_task: dict[str, list[bool]] = {}
    for trial in trials:
        if not trial.invalid:
            by_task.setdefault(trial.task, []).append(trial.resolved)
    return Standing(
        arm=arm,
        runs=len({trial.job for trial in trials}),
        rate=_rate([(sum(found), len(found)) for found in by_task.values()]),
        margin=margin(by_task),
        pass_at_k=sum(any(found) for found in by_task.values()) / len(by_task),
    )


def _read(directory: Path, lane: str) -> dict[str, list[Trial]]:
    """
    Every trial of each arm under `results/<lane>/`, by arm, raising
    LeaderboardError when there is no arm, for an arm with no trials or with
    only invalid ones, and for the first trial with a patch whose verifier
    applied none of it or could not apply it, so that nothing is ranked that
    cannot be.
    """
    lane_directory = Path(directory) / "results" / lane
    # Every directory: an arm's name may start with a `.`, which marks Harbor's own directories only inside a job.
    found = lane_directory.iterdir() if lane_directory.is_dir() else []
    arms = sorted(entry.name for entry in found if entry.is_dir())
    if not arms:
        raise LeaderboardError(f"no results under results/{lane}/")
    trials = {}
    for arm in arms:
        trials[arm] = read_trials(directory, arm, lane)
        if not trials[arm]:
            raise LeaderboardError(f"no results for arm {arm} under results/{lane}/")
        for trial in trials[arm]:
            grading = f"verifier in {trial.job}"
            # That verifier did not grade the patch, so its 0 is not the trial's result.
            if trial.patch_bytes > 0 and trial.verifier_applied is False:
                raised = f"raised {trial.verifier_raised} and " if trial.verifier_raised else ""
                raise LeaderboardError(
                    f"the trial of {trial.task} in results/{lane}/{arm}/{trial.job} made a "
                    f"{trial.patch_bytes}-byte patch, but its {grading} {raised}applied none"
                )
            # Its result is the grade of a change the harness did not make.
            if trial.verifier_refused:
                raise LeaderboardError(
                    f"the trial of {trial.task} in results/{lane}/{arm}/{trial.job} made a "
                    f"patch, but its {grading} could not apply it"
                )
        # Such an arm has no rate to print.
        if all(trial.invalid for trial in trials[arm]):
            raise LeaderboardError(
                f"every trial of arm {arm} under results/{lane}/ is invalid: "
                "its verifier wrote an empty reward"
            )
    return trials


def render(
    standings: Sequence[Standing],
    untested: Sequence[tuple[str, str, str]] = (),
    invalid: Sequence[tuple[str, str, str]] = (),
    unjudged: Sequence[tuple[str, str, str]] = (),
) -> str:
    """
    The table of `standings`, in the order given, then each (arm, job, task)
    in `untested`, then in `unjudged`, then in `invalid`, each under its own
    heading, unless there are none.
    """
    header = ["arm", "runs", "resolved rate", "pass@k"]
    rows = [
        [
            s.arm,
            str(s.runs),
            f"{_whole(s.rate)}% ±{_whole(s.margin)}",
            f"{_whole(s.pass_at_k)}% @{s.runs}",
        ]
        for s in standings
    ]
    lines = _table(header, rows)
    lines += _named(
        "These trials' verifiers recorded no test result, so their grading may have broken:", untested
    )
    lines += _named(
        "These trials' judges gave no verdict on some rubric item, so their grading may have broken:",
        unjudged,
    )
    lines += _named(
        "These trials' verifiers judged them invalid, so the leaderboard counts none of them:", invalid
    )
    return "\n".join(lines) + "\n"


def _named(heading: str, trials: Sequence[tuple[str, str, str]]) -> list[str]:
    """The lines naming each (arm, job, task) under the heading, or none when there are none."""
    if not trials:
        return []
    arm_width = max(len(arm) for arm, _, _ in trials)
    job_width = max(len(job) for _, job, _ in trials)
    return ["", heading] + [
        f"  {arm.ljust(arm_width)}  {job.ljust(job_width)}  {task}" for arm, job, task in trials
    ]


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """The rows under the header, the first column to the left and the rest to the right."""
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]
    return [
        "  ".join(
            [row[0].ljust(widths[0])] + [cell.rjust(width) for cell, width in zip(row[1:], widths[1:])]
        )
        for row in [header, *rows]
    ]


def _whole(share: float) -> int:
    """The share as a whole percent, a half rounded up."""
    return int(share * 100 + 0.5)


@dataclass(frozen=True)
class Problem:
    """
    One trial of a slopcodebench problem: each checkpoint it declares, in
    order, with what that checkpoint's verifier gave.

    A checkpoint never graded, as one after the problem stopped, has None for
    every measure. Erosion and verbosity are also None where scb-check
    measured nothing, whose 0.0 in the reward is not a score.
    """

    job: str
    task: str
    strict: tuple[float | None, ...]
    core: tuple[float | None, ...]
    erosion: tuple[float | None, ...]
    verbosity: tuple[float | None, ...]

    @property
    def graded(self) -> int:
        return sum(value is not None for value in self.strict)


@dataclass(frozen=True)
class QualityStanding:
    """An arm's row of the slopcodebench leaderboard. A measure nothing gave is None."""

    arm: str
    runs: int
    strict: float
    margin: float
    core: float
    erosion: float | None
    verbosity: float | None
    # The share of transitions between measured checkpoints where erosion went up.
    erosion_rose: float | None


def read_problems(directory: Path, arm: str) -> list[Problem]:
    """
    Every trial in the arm's agent jobs under `results/slopcodebench/<arm>/`,
    in `directory`, sorted by job and task.

    Raises LeaderboardError for a trial of a problem revision 4 does not have,
    whose checkpoints the leaderboard cannot count.
    """
    arm_directory = Path(directory) / "results" / lanes.SLOPCODEBENCH / arm
    if not arm_directory.is_dir():
        return []
    problems = []
    for job in sorted(_directories(arm_directory)):
        for trial in _directories(job):
            task = _task(trial)
            if task is None:
                continue
            declared = lanes.slopcodebench.CHECKPOINTS.get(task)
            if declared is None:
                raise LeaderboardError(
                    f"the trial of {task} in results/{lanes.SLOPCODEBENCH}/{arm}/{job.name} is of a problem"
                    " revision 4 does not have, so its checkpoints cannot be counted"
                )
            problems.append(_problem(job.name, task, trial, declared))
    return sorted(problems, key=lambda problem: (problem.job, problem.task))


def _problem(job: str, task: str, trial: Path, declared: int) -> Problem:
    result = _json(trial / "result.json")
    found = result.get("step_results") if isinstance(result, dict) else None
    steps = {step.get("step_name"): step for step in found or [] if isinstance(step, dict)}
    strict, core, erosion, verbosity = [], [], [], []
    for n in range(1, declared + 1):
        name = f"checkpoint_{n}"
        verifier = (steps.get(name) or {}).get("verifier_result")
        rewards = verifier.get("rewards") if isinstance(verifier, dict) else None
        rewards = rewards if isinstance(rewards, dict) else {}
        graded = _number(rewards.get("strict_pass_rate")) is not None and _number(rewards.get("core_pass_rate")) is not None
        details = _json(trial / "steps" / name / "verifier" / "reward_details.json")
        measured = graded and isinstance(details, dict) and details.get("scb_check_rc") == 0
        strict.append(_number(rewards.get("strict_pass_rate")) if graded else None)
        core.append(_number(rewards.get("core_pass_rate")) if graded else None)
        erosion.append(_number(rewards.get("erosion")) if measured else None)
        verbosity.append(_number(rewards.get("verbosity")) if measured else None)
    return Problem(job, task, tuple(strict), tuple(core), tuple(erosion), tuple(verbosity))


def _number(value) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def quality_report(directory: Path) -> str:
    """
    What `swe-drill leaderboard --lane slopcodebench` prints, run in `directory`.

    Each trial's pass rates are over every checkpoint its problem declares, one
    never graded counting 0, and its erosion and verbosity over those scb-check
    measured. Each problem weighs the same, its runs averaged first. The table
    goes by strict pass rate, best first, then by arm name. After it come the
    trials with a checkpoint never graded, then the checkpoints scb-check
    measured nothing of, each list by arm in the table's order, then by job and
    task.

    Raises ConfigError as `report` does, and LeaderboardError when no arm has
    results, an arm has no trials, or a trial is of a problem revision 4 does
    not have.
    """
    config.check_arms(Path(directory) / "swe-drill.toml")
    lane_directory = Path(directory) / "results" / lanes.SLOPCODEBENCH
    found = lane_directory.iterdir() if lane_directory.is_dir() else []
    arms = sorted(entry.name for entry in found if entry.is_dir())
    if not arms:
        raise LeaderboardError(f"no results under results/{lanes.SLOPCODEBENCH}/")
    problems = {}
    for arm in arms:
        problems[arm] = read_problems(directory, arm)
        if not problems[arm]:
            raise LeaderboardError(f"no results for arm {arm} under results/{lanes.SLOPCODEBENCH}/")
    standings = sorted(
        (_quality_standing(arm, found) for arm, found in problems.items()),
        key=lambda standing: (-standing.strict, standing.arm),
    )
    order = [standing.arm for standing in standings]
    stopped = [
        (arm, problem.job, f"{problem.task}  graded {problem.graded} of {len(problem.strict)}")
        for arm in order
        for problem in problems[arm]
        if problem.graded < len(problem.strict)
    ]
    unmeasured = [
        (arm, problem.job, f"{problem.task}  checkpoint_{n}")
        for arm in order
        for problem in problems[arm]
        for n, (strict, erosion) in enumerate(zip(problem.strict, problem.erosion), start=1)
        if strict is not None and erosion is None
    ]
    header = ["arm", "runs", "strict pass rate", "core pass rate", "erosion", "verbosity", "erosion rose"]
    rows = [
        [
            s.arm,
            str(s.runs),
            f"{_whole(s.strict)}% ±{_whole(s.margin)}",
            f"{_whole(s.core)}%",
            "—" if s.erosion is None else f"{s.erosion:.2f}",
            "—" if s.verbosity is None else f"{s.verbosity:.2f}",
            "—" if s.erosion_rose is None else f"{_whole(s.erosion_rose)}%",
        ]
        for s in standings
    ]
    lines = _table(header, rows)
    lines += _named(
        "These trials have checkpoints that were never graded, each counting 0:", stopped
    )
    lines += _named(
        "scb-check measured nothing of these checkpoints, so erosion and verbosity leave them out:", unmeasured
    )
    return "\n".join(lines) + "\n"


def _quality_standing(arm: str, problems: Sequence[Problem]) -> QualityStanding:
    def by_task(score) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for problem in problems:
            value = score(problem)
            if value is not None:
                found.setdefault(problem.task, []).append(value)
        return found

    def over_all(values: Sequence[float | None]) -> float:
        return sum(value or 0.0 for value in values) / len(values)

    def over_measured(values: Sequence[float | None]) -> float | None:
        measured = [value for value in values if value is not None]
        return statistics.fmean(measured) if measured else None

    strict = by_task(lambda problem: over_all(problem.strict))
    rises = [
        later > earlier
        for problem in problems
        for earlier, later in zip(problem.erosion, problem.erosion[1:])
        if earlier is not None and later is not None
    ]
    return QualityStanding(
        arm=arm,
        runs=len({problem.job for problem in problems}),
        strict=_mean_of_means(strict),
        margin=_mean_margin(strict),
        core=_mean_of_means(by_task(lambda problem: over_all(problem.core))),
        erosion=_mean_of_means(by_task(lambda problem: over_measured(problem.erosion))),
        verbosity=_mean_of_means(by_task(lambda problem: over_measured(problem.verbosity))),
        erosion_rose=sum(rises) / len(rises) if rises else None,
    )


def _mean_of_means(by_task: Mapping[str, Sequence[float]]) -> float | None:
    """The mean over tasks of each task's mean, so each task weighs the same. None for no task."""
    return statistics.fmean(statistics.fmean(found) for found in by_task.values()) if by_task else None


def _mean_margin(by_task: Mapping[str, Sequence[float]]) -> float:
    """
    Half the width of a 95% interval on `_mean_of_means`, from the same
    bootstrap `margin` uses: tasks drawn with replacement, each with every run.
    """
    means = [statistics.fmean(found) for found in by_task.values()]
    rng = random.Random(SEED)
    resampled = [statistics.fmean(rng.choices(means, k=len(means))) for _ in range(RESAMPLES)]
    cuts = statistics.quantiles(resampled, n=40, method="inclusive")
    return (cuts[-1] - cuts[0]) / 2
