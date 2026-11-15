"""Execute or verify exactly one frozen paper unit, checkpointing acceptance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_paper_simulation import sha256

STATE = Path("docs/PAPER_TASK_STATE.json")
PLAN = Path("docs/paper/run-plan.json")


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def verify_sources(plan):
    for filename, expected in plan["source_sha256"].items():
        if sha256(filename) != expected:
            raise ValueError("Frozen source changed: " + filename)


def verify(unit):
    root = Path(unit["output"])
    manifest = read(root / "manifest.json")
    if manifest["status"] != "complete":
        raise ValueError("Pipeline is not complete")
    data = Path(unit["data_directory"])
    generated = read(data / "manifest.json")
    for name, expected in generated["file_hashes"].items():
        if sha256(data / name) != expected:
            raise ValueError("Generated artifact changed: " + name)
    if manifest["dataset_sha256"] != generated["file_hashes"]["data.csv"]:
        raise ValueError("Fit data hash mismatch")
    model_root = root / f"random_seed_{unit['training_seed']}" / unit["model"]
    complete = read(model_root / "complete.json")
    if (complete["model"], complete["seed"], complete["fold"]) != (unit["model"], unit["training_seed"], "random"):
        raise ValueError("Completion identity mismatch")
    if set(complete["artifact_sha256"]) != set(unit["expected_model_artifacts"]):
        raise ValueError("Completion artifact set mismatch")
    for name, expected in complete["artifact_sha256"].items():
        if sha256(model_root / name) != expected:
            raise ValueError("Completed artifact changed: " + name)
    for name, expected in manifest["source_sha256"].items():
        if sha256(root / "source" / name) != expected or sha256(name) != expected:
            raise ValueError("Fit source snapshot mismatch: " + name)
    evaluation = read(model_root / "evaluation.json")
    membership = evaluation["split_rows"]
    if any(len(ids) != len(set(ids)) for ids in membership.values()):
        raise ValueError("Duplicate split rows")
    if any(set(membership[first]) & set(membership[second])
           for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Split leakage")
    return manifest, evaluation, model_root, generated


def run(unit_id, verify_only=False):
    plan = read(PLAN)
    verify_sources(plan)
    unit = next(item for item in plan["units"] if item["id"] == unit_id)
    record_path = Path(unit["output"]) / "paper-unit.json"
    if record_path.exists():
        verify(unit)
        record = read(record_path)
        if record["plan_sha256"] != sha256(PLAN) or record["unit"] != unit:
            raise ValueError("Unit plan identity mismatch")
        for filename, expected in record["acceptance_sha256"].items():
            if sha256(filename) != expected:
                raise ValueError("Accepted unit changed: " + filename)
        print("VERIFIED", unit_id, flush=True)
        return
    if verify_only:
        raise ValueError("Unit lacks paper acceptance manifest")
    state = read(STATE)
    state["current_task"] = "P13"
    state["tasks"]["P13"]["status"] = "in_progress"
    matching = [item for item in state["run_units"] if item["id"] == unit_id]
    if matching:
        entry = matching[0]
    else:
        entry = {"id": unit_id, "output": unit["output"]}
        state["run_units"].append(entry)
    entry.update(status="in_progress", started_at=datetime.now(timezone.utc).isoformat())
    save(STATE, state)
    data_directory = Path(unit["data_directory"])
    dataset = next(item for item in plan["datasets"] if item["output"] == str(data_directory))
    if not data_directory.exists():
        subprocess.run(dataset["generate_command"], check=True)
        subprocess.run(dataset["check_command"], check=True)
    else:
        generated = read(data_directory / "manifest.json")
        if (generated["seed"], generated["mechanism"], generated["rows"]) != (unit["generator_seed"], unit["mechanism"], 3000):
            raise ValueError("Existing generator identity mismatch")
        for filename, expected in generated["file_hashes"].items():
            if sha256(data_directory / filename) != expected:
                raise ValueError("Existing generator artifact changed")
    root = Path(unit["output"])
    command = unit["command"]
    if root.exists():
        verify(unit)
        elapsed = None
    else:
        log = Path("docs/paper/run-logs") / (unit_id + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with log.open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        if result.returncode:
            state = read(STATE)
            next(item for item in state["run_units"] if item["id"] == unit_id).update(status="failed", log=str(log), returncode=result.returncode)
            save(STATE, state)
            raise RuntimeError(f"Unit {unit_id} failed; inspect {log}, preserve attempt")
    manifest, evaluation, model_root, generated = verify(unit)
    membership = evaluation["split_rows"]
    split_hash = {name: hashlib.sha256(json.dumps(ids).encode()).hexdigest() for name, ids in membership.items()}
    state = read(STATE)
    for other in state["run_units"]:
        if other["id"] != unit_id and other.get("status") == "done":
            previous = read(Path(other["output"]) / "paper-unit.json")
            if previous["unit"]["data_directory"] == unit["data_directory"]:
                if previous["split_sha256"]["test"] != split_hash["test"]:
                    raise ValueError("Outer test membership differs across procedures")
                if previous["unit"]["model"] != "complete_case" and unit["model"] != "complete_case":
                    if previous["split_sha256"] != split_hash:
                        raise ValueError("Outer training/validation membership differs")
    parameter_counts = {"model": None, "loss": None}
    if (model_root / "model.pt").exists():
        from src.checkpoint import load_model_bundle
        bundle = load_model_bundle(model_root / "model.pt", data_path=data_directory / "data.csv")
        parameter_counts = {key: sum(value.numel() for value in bundle[name].parameters())
                            for key, name in (("model", "model"), ("loss", "loss_fn"))}
    elif unit["model"].endswith("_mlp"):
        import joblib
        model = joblib.load(model_root / "model.joblib")["model"]
        parameter_counts["model"] = sum(value.numel() for value in model.parameters())
    record = {"unit": unit, "kind": "frozen_synthetic_simulation", "status": "complete",
              "accepted_at": datetime.now(timezone.utc).isoformat(), "wall_seconds": elapsed,
              "plan_sha256": sha256(PLAN), "generator_manifest_sha256": sha256(data_directory / "manifest.json"),
              "split_sha256": split_hash, "split_counts": {name: len(ids) for name, ids in membership.items()},
              "parameter_counts": parameter_counts, "selection": evaluation["selection"],
              "data_sha256": generated["file_hashes"],
              "acceptance_sha256": {str(path): sha256(path) for path in (root / "manifest.json", model_root / "complete.json")}}
    save(record_path, record)
    note = Path("docs/paper/task-notes") / ("P13-" + unit_id + ".md")
    note.write_text(f"# P13 {unit_id}\n\nCompleted `{unit['output']}`.\nExact command: `{' '.join(command)}`.\n\nPipeline, generator/source/artifact hashes, completion identity, split isolation\nand paired test membership verified; parameter counts saved in paper-unit.json.\nWall seconds: {elapsed}. No final endpoint used for selection.\n\nVerification: `.venv/bin/python scripts/run_paper_unit.py --unit {unit_id} --verify-only`.\nNext: next pending frozen unit; P14 only after all 150 accepted.\n")
    state = read(STATE)
    entry = next(item for item in state["run_units"] if item["id"] == unit_id)
    entry.update(status="done", outputs=[str(record_path), str(note)], wall_seconds=elapsed)
    completed = sum(item["status"] == "done" for item in state["run_units"])
    state["tasks"]["P13"].update(summary=f"{completed}/{len(plan['units'])} frozen units accepted; no aggregate endpoint inspected.", outputs=["docs/paper/run-plan.json", "docs/paper/run-logs", "docs/paper/task-notes/P13-*.md"])
    state["last_handoff"] = str(note)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    if completed == len(plan["units"]):
        state["tasks"]["P13"].update(status="done", next_task="P14")
        state["current_task"] = "P14"
    save(STATE, state)
    print("ACCEPTED", unit_id, f"{completed}/{len(plan['units'])}", elapsed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    run(arguments.unit, arguments.verify_only)
