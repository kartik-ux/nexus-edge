"""Simple local model registry with versioned files and active pointer."""
from pathlib import Path
import json
import shutil
from datetime import datetime, timezone
from .model import save_model, load_model

class ModelRegistry:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_file = self.root / "active.json"

    def active_version(self):
        if not self.active_file.exists():
            return None
        return json.loads(self.active_file.read_text())["version"]

    def active_model_path(self):
        version = self.active_version()
        return self.root / version / "model.joblib" if version else None

    def load_active(self):
        path = self.active_model_path()
        return load_model(path) if path and path.exists() else None

    def promote(self, model, metrics, notes=""):
        version = datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%S%fZ")
        folder = self.root / version
        folder.mkdir(parents=True, exist_ok=False)
        save_model(model, folder / "model.joblib")
        metadata = {"version": version, "created_at": datetime.now(timezone.utc).isoformat(),
                    "metrics": metrics, "notes": notes}
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
        self.active_file.write_text(json.dumps({"version": version}, indent=2))
        return version

    def rollback(self, version):
        folder = self.root / version
        if not (folder / "model.joblib").exists():
            raise FileNotFoundError(f"Model version not found: {version}")
        self.active_file.write_text(json.dumps({"version": version}, indent=2))
