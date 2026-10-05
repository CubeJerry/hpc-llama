"""Private session state and the persistent sbatch / attachable srun lifecycle."""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import stat
import subprocess
import time
from typing import Any

from .contracts import AppError, ENDED_SCHEDULER_STATES, InferenceSettings, ModelSpec, ResourceRequest, SessionManifest, SiteProfile
from .schedulers import current_job_id, inside_allocation, scheduler_for


def private_dir(path: Path) -> Path:
    """Create an owner-only directory, rejecting symlink and ownership surprises."""
    path = Path(path).absolute()
    for parent in [*reversed(path.parents), path]:
        if parent.is_symlink():
            raise AppError("permission", "Session paths must not contain symbolic links")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise AppError("permission", "Session directory must belong to the current user")
    path.chmod(0o700)
    return path


def atomic_json(path: Path, value: Any) -> None:
    """Single-writer replace; never follow an existing state-file symlink."""
    private_dir(path.parent)
    if path.is_symlink():
        raise AppError("permission", "Refusing a symbolic link in session state")
    if path.exists() and path.stat().st_uid != os.getuid():
        raise AppError("permission", "State file must belong to the current user")
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(serialized.encode("utf-8")) > 64 * 1024 * 1024:
        raise AppError("storage", "Private session state reached its 64 MiB limit. Export chats and start a new session or select fewer attachments")
    temp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except OSError as exc:
        raise AppError("storage", "Could not save session state; check free space and permissions") from exc
    finally:
        temp.unlink(missing_ok=True)


def read_private_json(path: Path) -> dict:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            info = os.fstat(handle.fileno())
            if info.st_uid != os.getuid() or info.st_mode & 0o077 or not stat.S_ISREG(info.st_mode):
                raise AppError("permission", "Session state must be an owner-only regular file")
            if info.st_size > 64 * 1024 * 1024:
                raise AppError("storage", "Session snapshot exceeds the 64 MiB safety limit")
            return json.load(handle)
    except AppError:
        raise
    except (OSError, ValueError) as exc:
        raise AppError("storage", "Could not read session state") from exc


def private_token(path: Path) -> str:
    if not path.exists():
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(secrets.token_urlsafe(32))
        except FileExistsError:
            pass
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd) as stream:
            info = os.fstat(stream.fileno())
            if info.st_uid != os.getuid() or info.st_mode & 0o077 or not stat.S_ISREG(info.st_mode):
                raise AppError("permission", "Credential file must be private and owned by you")
            token = stream.read(256).strip()
        if len(token) < 32:
            raise AppError("auth", "Session credential file is invalid")
        return token
    except OSError as exc:
        raise AppError("permission", "Cannot safely read session credentials") from exc


def scheduler_environment() -> dict[str, str]:
    """Discard inherited resource overrides while preserving allocation identity."""
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("SBATCH_", "SRUN_")) and key not in {
                "SLURM_NTASKS", "SLURM_NPROCS", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_CPU",
                "SLURM_MEM_PER_NODE", "SLURM_GPUS", "SLURM_GPUS_PER_TASK", "SLURM_GPUS_PER_NODE",
                "SLURM_TRES_PER_TASK", "SLURM_NTASKS_PER_NODE", "SLURM_NNODES", "PBS_DPREFIX", "PBS_QSUB_SLEEP"}}


def _run(argv: list[str], timeout: int = 20) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False, env=scheduler_environment())
    except FileNotFoundError as exc:
        raise AppError("scheduler", f"{argv[0]} is unavailable. Use --demo here or install on a supported scheduler login node") from exc
    except subprocess.TimeoutExpired as exc:
        raise AppError("scheduler", f"{argv[0]} timed out; the allocation state is unknown") from exc


def walltime_seconds(value: str) -> int:
    days, clock = (value.split("-", 1) if "-" in value else ("0", value))
    hours, minutes, seconds = map(int, clock.split(":"))
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


class SessionManager:
    def __init__(self, state_root: Path, launcher: Path):
        self.root = private_dir(Path(state_root))
        self.sessions = private_dir(self.root / "sessions")
        self.launcher = Path(launcher).absolute()

    def create(self, model: ModelSpec, resources: ResourceRequest, profile: SiteProfile,
               settings: InferenceSettings, demo: bool = False) -> SessionManifest:
        if resources.gpu_count != 1:
            raise AppError("unsupported", "This release supports one GPU per session")
        manifest = SessionManifest(owner=getpass.getuser(), directory="", model=model,
                                   resources=resources, profile=profile, settings=settings, demo=demo)
        folder = private_dir(self.sessions / manifest.id)
        manifest.directory = str(folder)
        private_dir(folder / "logs")
        private_token(folder / "supervisor.key")
        atomic_json(folder / "manifest.json", manifest.model_dump())
        return manifest

    def load(self, session_id: str) -> SessionManifest:
        if not re.fullmatch(r"[0-9a-f]{32}", session_id):
            raise AppError("validation", "Invalid session ID")
        directory = self.sessions / session_id
        private_dir(directory)
        manifest = SessionManifest.model_validate(read_private_json(directory / "manifest.json"))
        if manifest.id != session_id or manifest.owner != getpass.getuser() or Path(manifest.directory) != directory:
            raise AppError("permission", "Session identity or ownership does not match")
        receipt = directory / "submission.json"
        if receipt.exists() and not manifest.job_id:
            submitted = read_private_json(receipt)
            if submitted.get("nonce") == manifest.nonce:
                manifest.job_id = submitted.get("job_id")
                manifest.cluster = submitted.get("cluster")
                manifest.scheduler_state = "PENDING"
        return manifest

    def list(self) -> list[SessionManifest]:
        result = []
        for path in sorted(self.sessions.iterdir(), reverse=True):
            if re.fullmatch(r"[0-9a-f]{32}", path.name):
                try:
                    result.append(self.load(path.name))
                except AppError:
                    continue
        return sorted(result, key=lambda item: item.created_at, reverse=True)

    def render_batch(self, manifest: SessionManifest) -> str:
        return scheduler_for(manifest.profile).render_batch(manifest, self.launcher, self.root)

    def submit(self, manifest: SessionManifest) -> SessionManifest:
        manifest = self.load(manifest.id)
        if manifest.job_id or manifest.supervisor_pid:
            raise AppError("busy", "This session has already been started. Choose Resume")
        folder = Path(manifest.directory)
        if manifest.demo:
            logfile = os.open(folder / "logs" / "supervisor.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            try:
                subprocess.Popen([str(self.launcher), "--state-root", str(self.root), "_supervisor", manifest.id],
                                 stdin=subprocess.DEVNULL, stdout=logfile, stderr=logfile,
                                 start_new_session=True, cwd=self.launcher.parent)
            finally:
                os.close(logfile)
            return manifest
        script = folder / "allocation.sh"
        if script.is_symlink():
            raise AppError("permission", "Unsafe allocation script path")
        script_content = self.render_batch(manifest)
        fd = os.open(script, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as out:
            out.write(script_content)
        try:
            job_id, cluster = scheduler_for(manifest.profile).submit(script, manifest, _run)
        except AppError:
            # Leave inspectable submission material after ambiguous qsub/sbatch
            # replies. No automatic retry can accidentally create a second job.
            raise
        atomic_json(folder / "submission.json", {"job_id": job_id, "cluster": cluster, "nonce": manifest.nonce})
        manifest.job_id, manifest.cluster, manifest.scheduler_state = job_id, cluster, "PENDING"
        return manifest

    def restart_saved(self, manifest: SessionManifest, resources: ResourceRequest | None = None,
                      profile: SiteProfile | None = None) -> SessionManifest:
        """Explicit new allocation with immutable saved content, never replayed work."""
        previous = self.status(manifest)
        scheduler = (previous.scheduler_state.split() or [""])[0].rstrip("+")
        if previous.demo:
            if previous.backend_state not in {"stopped", "failed"}:
                raise AppError("busy", "Detach leaves the session running. Stop it before starting a new session from its saved chats")
        elif previous.job_id and scheduler not in ENDED_SCHEDULER_STATES:
            raise AppError("busy", "The previous allocation is still active or its status is unknown. Resume it or verify it has ended before restarting")
        elif not previous.job_id and previous.backend_state not in {"stopped", "failed"}:
            raise AppError("busy", "The previous session has not finished")
        original_path = Path(previous.directory) / "snapshot.json"
        if not original_path.exists():
            raise AppError("storage", "This session has no saved conversation snapshot")
        snapshot = read_private_json(original_path)
        if snapshot.get("session_id") != previous.id or snapshot.get("nonce") != previous.nonce:
            raise AppError("auth", "Saved conversation identity does not match its session")
        requested_resources = (resources or previous.resources).model_copy(deep=True)
        effective_settings = previous.settings.model_copy(deep=True)
        # A restarted allocation may have fewer CPUs; retain one for services.
        effective_settings.threads = min(effective_settings.threads, requested_resources.cpus - 1)
        effective_settings.threads_batch = min(effective_settings.threads_batch, requested_resources.cpus - 1)
        fresh = self.create(previous.model.model_copy(deep=True), requested_resources,
                            (profile or previous.profile).model_copy(deep=True), effective_settings, demo=previous.demo)
        snapshot["session_id"], snapshot["nonce"] = fresh.id, fresh.nonce
        max_web_id = 0
        for chat in snapshot.get("conversations", {}).values():
            for turn in chat.get("turns", []):
                if turn.get("status") in {"running", "awaiting_approval"}:
                    turn["status"] = "interrupted"
                    turn["error"] = "Previous allocation ended. This response was not replayed; Retry is an explicit new request"
            for source_id in chat.get("sources", {}):
                if re.fullmatch(r"W[0-9]+", source_id):
                    max_web_id = max(max_web_id, int(source_id[1:]))
        for approval in snapshot.get("approvals", {}).values():
            if approval.get("status") in {"pending", "approved"}:
                approval["status"] = "expired"
                approval["error"] = "The previous allocation ended. Outbound actions require a new request and approval"
        for attachment in snapshot.get("attachments", {}).values():
            attachment["snapshot_path"] = None  # Serialized immutable content is authoritative.
        atomic_json(Path(fresh.directory) / "snapshot.json", snapshot)
        web_directory = private_dir(Path(fresh.directory) / "web")
        fd = os.open(web_directory / "source-sequence", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(str(max_web_id))
        return fresh

    def inventory(self, profile: SiteProfile | None = None) -> dict:
        """Cached read-only inventory; unavailable schedulers use the site profile."""
        profile = profile or SiteProfile()
        cache = self.root / ("inventory.json" if profile.scheduler == "slurm" else "inventory-pbs.json")
        if cache.exists():
            try:
                cached = read_private_json(cache)
                if 0 <= time.time() - cached.get("observed_at", 0) < 300:
                    return cached
            except AppError:
                pass
        try:
            inventory = scheduler_for(profile).inventory(profile, _run)
        except AppError:
            inventory = {"available": False, "partitions": [], "scheduler": profile.scheduler}
        inventory["observed_at"] = time.time()
        if not inventory["available"]:
            inventory["error"] = f"Read-only {profile.scheduler.upper()} inventory unavailable; use the editable site profile"
        atomic_json(cache, inventory)
        return inventory

    def status(self, manifest: SessionManifest) -> SessionManifest:
        manifest = self.load(manifest.id)
        if manifest.demo or not manifest.job_id:
            return manifest
        return scheduler_for(manifest.profile).status(manifest, _run)

    def attach_argv(self, manifest: SessionManifest) -> list[str]:
        command = [str(self.launcher), "--state-root", str(self.root), "_tui", manifest.id]
        if manifest.demo or (manifest.profile.scheduler == "slurm" and inside_allocation(manifest)):
            return command
        return scheduler_for(manifest.profile).attach_argv(manifest, self.launcher, self.root, _run)

    def stop(self, manifest: SessionManifest) -> None:
        fresh = self.load(manifest.id)
        if fresh.nonce != manifest.nonce or fresh.owner != getpass.getuser():
            raise AppError("permission", "Session identity does not match")
        if fresh.demo:
            raise AppError("unsupported", "Stop a demo through its authenticated supervisor API")
        if not fresh.job_id:
            raise AppError("scheduler", "Session does not have a submitted allocation")
        checked = self.status(fresh)
        scheduler_for(checked.profile).stop(checked, _run)
