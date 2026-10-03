"""Hash final delivery files and create a local archive after validation."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages" / "codex_r2op_reviewed"


def main():
    checks = json.loads((ROOT / "experiments" / "codex" / "package_validation.json").read_text(encoding="utf-8"))
    manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))
    evidence = json.loads((ROOT / "experiments" / "codex" / "evidence.json").read_text(encoding="utf-8"))
    assert checks["passed"] and not any(row["accepted"] for row in evidence["gate"])
    manifest["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PACKAGE.iterdir()) if p.is_file() and p.name != "manifest.json"}
    assert checks["model_sha256"] == manifest["files_sha256"]["model.pkl"]
    assert checks["notebook_sha256"] == manifest["files_sha256"]["prediction_notebook.ipynb"]
    manifest.update(fresh_training_reproduction_max_difference=evidence["fresh_full_fit_max_difference"],
                    isolated_notebook_cases=len(checks["isolated_notebook_cases"]),
                    required_uploads=["prediction_notebook.ipynb", "model.pkl", "requirements.txt", "technical_report.pdf"])
    (PACKAGE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    target = ROOT / "deliverables" / "Track2_R2op_reviewed.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(PACKAGE.iterdir()):
            if p.is_file():
                z.write(p, arcname="Track2_R2op_reviewed/" + p.name)
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
    print(f"Final delivery: {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
