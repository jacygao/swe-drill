import importlib.util
import json
from pathlib import Path


def load_script():
    path = Path(__file__).resolve().parent.parent / "scripts" / "azure_retry.py"
    spec = importlib.util.spec_from_file_location("azure_retry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


retry = load_script()

SPENT = {"n_input_tokens": 1000, "cost_usd": 0.1}


def failure(kind: str) -> dict:
    return {"exception_type": kind, "exception_message": "", "exception_traceback": ""}


def write_trial(tmp_path: Path, result: dict, files: dict[str, dict] = {}) -> Path:
    trial = tmp_path / "task__x1"
    trial.mkdir()
    (trial / "result.json").write_text(json.dumps(result))
    for name, value in files.items():
        (trial / name).parent.mkdir(parents=True, exist_ok=True)
        (trial / name).write_text(json.dumps(value))
    return trial


def step(name: str, error: str | None = None, agent_result: dict | None = None) -> dict:
    return {
        "step_name": name,
        "agent_result": agent_result,
        "verifier_result": None,
        "exception_info": None if error is None else failure(error),
    }


def test_a_single_step_trial_that_failed_setting_up_is_retried(tmp_path):
    trial = write_trial(tmp_path, {"exception_info": failure("EnvironmentStartTimeoutError"), "agent_result": None})

    assert retry.never_worked(trial)


def test_a_single_step_trial_whose_agent_spent_and_failed_is_not_retried(tmp_path):
    trial = write_trial(tmp_path, {"exception_info": failure("NonZeroAgentExitCodeError"), "agent_result": SPENT})

    assert not retry.never_worked(trial)


def test_a_multi_step_trial_whose_first_step_failed_before_the_agent_ran_is_retried(tmp_path):
    trial = write_trial(
        tmp_path,
        {"exception_info": None, "agent_result": None, "step_results": [step("checkpoint_1", "HealthcheckError")]},
    )

    assert retry.never_worked(trial)


def test_a_multi_step_trial_whose_first_agent_failed_having_spent_nothing_is_retried(tmp_path):
    trial = write_trial(
        tmp_path,
        {
            "exception_info": None,
            "agent_result": None,
            "step_results": [step("checkpoint_1", "NonZeroAgentExitCodeError")],
        },
    )

    assert retry.never_worked(trial)


def test_a_multi_step_trial_that_failed_after_an_earlier_step_worked_is_not_retried(tmp_path):
    trial = write_trial(
        tmp_path,
        {
            "exception_info": None,
            "agent_result": None,
            "step_results": [
                step("checkpoint_1", agent_result=SPENT),
                step("checkpoint_2", "HealthcheckError"),
            ],
        },
    )

    assert not retry.never_worked(trial)


def test_a_multi_step_trial_whose_agent_spent_only_by_its_record_is_not_retried(tmp_path):
    # Harbor leaves a step's agent_result empty when the agent raised; SWE-drill's record still counts the credits.
    trial = write_trial(
        tmp_path,
        {
            "exception_info": None,
            "agent_result": None,
            "step_results": [step("checkpoint_1", "NonZeroAgentExitCodeError")],
        },
        {"steps/checkpoint_1/agent/suss.json": {"usage": {"ai_credits": 12.5}}},
    )

    assert not retry.never_worked(trial)


def test_a_multi_step_trial_that_timed_out_in_its_first_step_is_not_retried(tmp_path):
    trial = write_trial(
        tmp_path,
        {"exception_info": None, "agent_result": None, "step_results": [step("checkpoint_1", "AgentTimeoutError")]},
    )

    assert not retry.never_worked(trial)


def test_a_multi_step_trial_that_failed_setting_up_before_any_step_is_retried(tmp_path):
    trial = write_trial(
        tmp_path,
        {"exception_info": failure("WorkspaceError"), "agent_result": None, "step_results": None},
    )

    assert retry.never_worked(trial)


def test_a_multi_step_trial_that_finished_every_step_is_not_retried(tmp_path):
    trial = write_trial(
        tmp_path,
        {
            "exception_info": None,
            "agent_result": None,
            "step_results": [step("checkpoint_1", agent_result=SPENT), step("checkpoint_2", agent_result=SPENT)],
        },
    )

    assert not retry.never_worked(trial)
