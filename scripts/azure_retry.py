"""
Run again, up to argv[1] times, each trial whose agent never worked on its task.
Run from the checkout on the VM.

Each such trial is set aside under ~/retried/<attempt>/, and its job resumed
with the environment SWE-drill gives the job's `harbor run`. A trial the resume
made no replacement for is put back, so it still counts as unresolved. Exits
1 when a resume failed or a trial was put back.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

from swe_drill import cli, config

# Harbor raises these before the agent starts on the task.
SETUP = {
    "AgentSetupTimeoutError",
    "ComposeLoadError",
    "EnvironmentStartTimeoutError",
    "HealthcheckError",
    "SandboxBuildFailedError",
}


def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def spent(trial):
    """Whether the agent spent anything, in the trial or in any of its steps."""
    result = read(trial / "result.json")
    found = [(result.get("agent_result") or {}), (read(trial / "agent" / "suss.json").get("usage") or {})]
    for step in result.get("step_results") or []:
        found.append(step.get("agent_result") or {})
        found.append(read(trial / "steps" / str(step.get("step_name")) / "agent" / "suss.json").get("usage") or {})
    values = [part.get(key) for part in found for key in ("n_input_tokens", "cost_usd", "ai_credits")]
    return any(isinstance(value, (int, float)) and value > 0 for value in values)


def never_worked(trial):
    result = read(trial / "result.json")
    # A step's failure is recorded on the step, and the trial records none of its own.
    errors = [result.get("exception_info")] + [step.get("exception_info") for step in result.get("step_results") or []]
    error = next(((info or {}).get("exception_type") for info in errors if info), None)
    if error is None or error == "AgentTimeoutError":
        return False
    # A later step can fail setting up after an earlier one did real work, which a retry would throw away.
    if result.get("step_results"):
        return not spent(trial)
    return error in SETUP or not spent(trial)


def trials(job):
    return [t for t in job.iterdir() if t.is_dir() and not t.name.startswith(".")]


def task(trial):
    """The task a trial ran: its directory's name less Harbor's suffix."""
    return trial.name.rsplit("__", 1)[0]


def main():
    retries, arm, lane = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    check = "--check" in sys.argv[4:]
    harbor = Path(sys.executable).parent / "harbor"
    settings = config.load(Path("swe-drill.toml"), arm, lane)
    job_command = cli._plan(settings, (), lane, "retry", [], check, harbor).jobs[0]
    env = cli._environment(job_command)
    failed = False
    for attempt in range(1, retries + 1):
        jobs = [path.parent for path in sorted(Path(".").glob("*/*/*/*/config.json"))]
        # Every trial is chosen before any moves, so a fault while choosing moves none.
        chosen = {
            job: [t for t in trials(job) if never_worked(t)]
            for job in jobs
            if job.parts[0] in ("results", "checks")
        }
        chosen = {job: found for job, found in chosen.items() if found}
        if not chosen:
            break
        aside = Path.home() / "retried" / str(attempt)
        for job, found in chosen.items():
            moved = []
            try:
                (aside / job).mkdir(parents=True, exist_ok=True)
                for trial in found:
                    shutil.move(str(trial), str(aside / trial))
                    moved.append(trial)
                print(f"azure-run: attempt {attempt} at {len(found)} trials in {job} whose agent never worked on the task", flush=True)
                if subprocess.call([str(harbor), "jobs", "resume", "-p", str(job)], env=env) != 0:
                    print(f"azure-run: harbor jobs resume failed on {job}", flush=True)
                    failed = True
            except Exception as error:
                print(f"azure-run: the retry of {job} failed: {error}", flush=True)
                failed = True
            finally:
                # Whatever failed, a set-aside trial with no replacement goes back.
                made = {task(t) for t in trials(job)}
                for trial in moved:
                    if task(trial) not in made:
                        shutil.move(str(aside / trial), str(trial))
                        print(f"azure-run: put back {trial.name}, which the resume did not run again", flush=True)
                        failed = True
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
