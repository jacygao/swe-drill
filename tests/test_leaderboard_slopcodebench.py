import json
from pathlib import Path

import pytest

from swe_drill import cli

LANE = "slopcodebench"
SWE_DRILL_TOML = """\
environment = "docker"

[arms.alpha]
agent = "swe_drill.agents.copilot_cli:CopilotCli"
"""


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def step(strict: float | None, core: float | None = None, erosion: float = 0.3, verbosity: float = 0.1, scored: bool = True):
    """A checkpoint's grade: None for a strict pass rate the verifier never gave."""
    return {"strict": strict, "core": strict if core is None else core, "erosion": erosion, "verbosity": verbosity, "scored": scored}


def write_trial(root: Path, arm: str, job: str, task: str, steps: list[dict], suffix: str = "x1") -> Path:
    """A trial of the problem as Harbor leaves one: a step result per checkpoint it reached."""
    trial = root / "results" / LANE / arm / job / f"{task}__{suffix}"
    results = []
    for n, given in enumerate(steps, start=1):
        name = f"checkpoint_{n}"
        rewards = None
        if given["strict"] is not None:
            # As the task's shim writes them, 0.0 for a measure scb-check gave none of.
            rewards = {
                "core_pass_rate": given["core"],
                "strict_pass_rate": given["strict"],
                "isolated_pass_rate": given["strict"],
                "erosion": given["erosion"] if given["scored"] else 0.0,
                "verbosity": given["verbosity"] if given["scored"] else 0.0,
            }
            write_json(trial / "steps" / name / "verifier" / "reward_details.json", {"scb_check_rc": 0 if given["scored"] else 1})
        results.append({"step_name": name, "verifier_result": None if rewards is None else {"rewards": rewards}})
    write_json(trial / "result.json", {"task_name": f"gabeorlanski/{task}", "step_results": results})
    return trial


def run_leaderboard(root: Path, capsys) -> tuple[int, str, str]:
    code = cli.main(["leaderboard", "--lane", LANE], directory=root)
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def root(tmp_path) -> Path:
    (tmp_path / "swe-drill.toml").write_text(SWE_DRILL_TOML)
    return tmp_path


def test_a_problem_scores_every_checkpoint_it_declares_with_one_never_graded_counting_0(root, capsys):
    # dynamic_buffer declares 4 checkpoints, and the problem stopped after its second.
    write_trial(root, "alpha", "S-1", "dynamic_buffer", [step(1.0, 0.5), step(0.5, 1.0)])

    code, out, err = run_leaderboard(root, capsys)

    assert (code, err) == (0, "")
    assert out == (
        "arm    runs  strict pass rate  core pass rate  erosion  verbosity  erosion rose\n"
        "alpha     1            38% ±0             38%     0.30       0.10            0%\n"
        "\n"
        "These trials have checkpoints that were never graded, each counting 0:\n"
        "  alpha  S-1  dynamic_buffer  graded 2 of 4\n"
    )


def test_each_problem_weighs_the_same_whatever_its_runs(root, capsys):
    # cfgpipe declares 6 and dag_execution 3.
    for job in ("S-1", "S-2"):
        write_trial(root, "alpha", job, "cfgpipe", [step(1.0)] * 6)
    write_trial(root, "alpha", "S-1", "dag_execution", [step(0.0)] * 3)

    code, out, err = run_leaderboard(root, capsys)

    assert (code, err) == (0, "")
    [line] = [line for line in out.splitlines() if line.startswith("alpha")]
    assert line.split()[:4] == ["alpha", "2", "50%", "±50"]


def test_erosion_and_verbosity_leave_out_a_checkpoint_scb_check_measured_nothing_of(root, capsys):
    write_trial(
        root,
        "alpha",
        "S-1",
        "dag_execution",
        [step(1.0, erosion=0.2, verbosity=0.1), step(1.0, scored=False), step(1.0, erosion=0.4, verbosity=0.3)],
    )

    code, out, err = run_leaderboard(root, capsys)

    assert (code, err) == (0, "")
    assert out == (
        "arm    runs  strict pass rate  core pass rate  erosion  verbosity  erosion rose\n"
        "alpha     1           100% ±0            100%     0.30       0.20             —\n"
        "\n"
        "scb-check measured nothing of these checkpoints, so erosion and verbosity leave them out:\n"
        "  alpha  S-1  dag_execution  checkpoint_2\n"
    )


def test_erosion_rose_is_the_share_of_measured_transitions_where_it_went_up(root, capsys):
    write_trial(
        root,
        "alpha",
        "S-1",
        "dynamic_buffer",
        [step(1.0, erosion=0.2), step(1.0, erosion=0.3), step(1.0, erosion=0.25), step(1.0, erosion=0.4)],
    )

    code, out, err = run_leaderboard(root, capsys)

    assert (code, err) == (0, "")
    [line] = [line for line in out.splitlines() if line.startswith("alpha")]
    assert line.split()[-1] == "67%"


def test_arms_go_by_strict_pass_rate_then_by_name(root, capsys):
    write_trial(root, "beta", "S-1", "dag_execution", [step(0.5)] * 3)
    write_trial(root, "alpha", "S-1", "dag_execution", [step(0.5)] * 3)
    write_trial(root, "gamma", "S-1", "dag_execution", [step(1.0)] * 3)

    code, out, err = run_leaderboard(root, capsys)

    assert (code, err) == (0, "")
    assert [line.split()[0] for line in out.splitlines()[1:4]] == ["gamma", "alpha", "beta"]


def test_a_problem_revision_4_does_not_have_is_an_error_naming_it(root, capsys):
    write_trial(root, "alpha", "S-1", "newcomer", [step(1.0)] * 3)

    code, out, err = run_leaderboard(root, capsys)

    assert code != 0 and out == ""
    assert "newcomer" in err and "revision 4" in err
