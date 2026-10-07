"""Launcher runs on login node; inference/files run only in an allocation (or DEMO)."""
from __future__ import annotations
import argparse
import builtins
import asyncio
import getpass
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from .contracts import AppError, InferenceSettings, ModelSpec, ResourceRequest, SiteProfile, session_is_ended

APP_ROOT = Path(__file__).resolve().parents[2]


def display(*values, **kwargs):
    # CLI diagnostics also treat filenames/provider errors as untrusted terminal text.
    from .files import clean_text
    builtins.print(*(clean_text(value) for value in values), **kwargs)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="hpc-llm", description="Local LLM sessions inside persistent Slurm or PBS allocations")
    parser.add_argument("--state-root", type=Path, default=Path(os.environ.get("HPC_LLM_HOME", APP_ROOT / "state")))
    parser.add_argument("--demo", action="store_true", help="CPU-only simulated backend; no Slurm, internet or model download")
    parser.add_argument("--check", action="store_true", help="Check installed application without allocating or networking")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor")
    sub.add_parser("sessions")
    attach = sub.add_parser("attach"); attach.add_argument("session_id", nargs="?")
    stop = sub.add_parser("stop"); stop.add_argument("session_id")
    for command in ("_tui", "_supervise", "_supervisor"):
        internal = sub.add_parser(command); internal.add_argument("session_id")
    models = sub.add_parser("models")
    models_sub = models.add_subparsers(dest="model_action")
    register = models_sub.add_parser("register"); register.add_argument("path"); register.add_argument("--projector"); register.add_argument("--mtp", help="Matching local MTP head GGUF")
    models_sub.add_parser("list", help="List registered models and IDs")
    remote = models_sub.add_parser("remote"); remote.add_argument("repo"); remote.add_argument("--revision", default="main")
    download = models_sub.add_parser("download"); download.add_argument("repo"); download.add_argument("filename"); download.add_argument("--revision", required=True)
    download.add_argument("--projector-filename", help="Download a matching projector from the same repository revision")
    download.add_argument("--mtp-filename", help="Download a matching MTP head from the same revision")
    install = models_sub.add_parser("install", help="Install a GGUF model or add companions to an existing model")
    install.add_argument("source", nargs="?", help="Hugging Face repository or model URL; optional for a registered model with source provenance")
    install.add_argument("--model", help="Registered model ID or path; download companions only")
    install.add_argument("--quant"); install.add_argument("--revision", help="Revision; existing model revision is retained when adding companions")
    install.add_argument("--projector", help="auto, none, or exact projector filename; omitted preserves existing companion")
    install.add_argument("--mtp", default="none", help="auto, none, or exact MTP head filename")
    install.add_argument("--verify-full", action="store_true", help="Recheck every byte instead of reusing unchanged-file verification")
    install.add_argument("--dry-run", action="store_true", help="Show pinned files and cache location without downloading weights")
    profiles = sub.add_parser("profiles", help="Manage site profiles, resource presets and model cache")
    ps = profiles.add_subparsers(dest="profile_action")
    ps.add_parser("list")
    show = ps.add_parser("show"); show.add_argument("name", nargs="?")
    use = ps.add_parser("use"); use.add_argument("name")
    cache = ps.add_parser("cache"); cache.add_argument("path")
    create = ps.add_parser("create"); create.add_argument("name"); create.add_argument("--base", default="generic")
    create.add_argument("--scheduler", choices=["slurm", "pbs"])
    create.add_argument("--pbs-gpu-resource"); create.add_argument("--pbs-gpu-type-resource")
    preset = ps.add_parser("preset"); preset.add_argument("profile"); preset.add_argument("name")
    for command in (create, preset):
        command.add_argument("--partition", "--queue", dest="partition")
        for field in ("gpu-type", "walltime", "account", "qos", "constraint"):
            command.add_argument("--" + field)
        for field in ("gpu-count", "cpus", "memory-gb"):
            command.add_argument("--" + field, type=int)
    smoke = sub.add_parser("smoke", help="WEHI smoke test: dry run unless --execute")
    smoke.add_argument("--model", required=True); smoke.add_argument("--execute", action="store_true")
    smoke.add_argument("--runtime", default=""); smoke.add_argument("--minutes", type=int, default=20)
    smoke.add_argument("--projector", help="Matching vision projector GGUF")
    smoke.add_argument("--mtp", help="Matching local MTP head GGUF")
    smoke.add_argument("--acceleration", choices=["off", "auto", "mtp"], default="off")
    smoke.add_argument("--memory-gb", type=int)
    smoke.add_argument("--gpu-type")
    return parser.parse_args(argv)


def profile_for(root: Path) -> SiteProfile:
    from .profiles import ProfileStore
    return ProfileStore(root, APP_ROOT).load()


def profiles_command(args, root):
    from .profiles import ProfileStore
    store = ProfileStore(root, APP_ROOT)
    action = args.profile_action
    if action in (None, "list"):
        active = store.config().get("profile", "wehi")
        for profile in store.list():
            display(f"{'*' if profile.name == active else ' '} {profile.name} ({profile.scheduler}) — {profile.description}")
        return 0
    if action == "show":
        result = store.load(args.name)
    elif action == "use":
        result = store.select(args.name)
    elif action == "cache":
        result = {"model_cache": str(store.set_cache(args.path))}
    else:
        base = args.base if action == "create" else args.profile
        # Preserve portable expressions in raw profile templates.
        profile = next((p for p in store.list() if p.name == base), None)
        if profile is None:
            profile = store.load(base)
        data = profile.model_dump()
        resources = dict(data["resources"])
        for field in ResourceRequest.model_fields:
            value = getattr(args, field, None)
            if value is not None:
                resources[field] = value
        resource = ResourceRequest.model_validate(resources)
        if action == "create":
            if any(p.name == args.name for p in store.list()):
                raise AppError("validation", "That profile already exists. Edit it in Site setup or choose a new name.")
            data.update(name=args.name, resources=resource.model_dump())
            for field in ("scheduler", "pbs_gpu_resource", "pbs_gpu_type_resource"):
                if getattr(args, field, None) is not None:
                    data[field] = getattr(args, field)
        else:
            data["resource_presets"][args.name] = resource.model_dump()
        result = store.save(SiteProfile.model_validate(data), select=action == "create")
    display(json.dumps(result.model_dump() if hasattr(result, "model_dump") else result, indent=2))
    return 0


def _manager(root):
    from .lifecycle import SessionManager
    return SessionManager(root.resolve(), (APP_ROOT / "hpc-llm").resolve())


def _library(root):
    from .models import ModelLibrary
    return ModelLibrary(root / "models", cache_dir=Path(profile_for(root).model_cache))


def check(root):
    from . import __version__
    modules = ["aiohttp", "textual", "pydantic", "pypdf", "python-docx", "openpyxl", "pillow", "beautifulsoup4", "ddgs"]
    display(f"HPC LLM {__version__} | Python {sys.version.split()[0]}")
    for module in modules:
        display(f"  {module}: {importlib.metadata.version(module)}")
    display("Application imports and configuration: OK")
    InferenceSettings(); ResourceRequest(); profile_for(root)
    display("No GPU allocated, no model loaded, no web request made.")
    return 0


def doctor(root):
    check(root)
    from .runtime import probe_runtime
    profile = profile_for(root)
    display(f"State: {root.resolve()}")
    display(f"Model cache: {profile.model_cache}")
    display(f"Profile: {profile.name} ({profile.scheduler})")
    commands = ("sbatch", "srun", "squeue", "sacct", "scancel") if profile.scheduler == "slurm" else ("qsub", "qstat", "qdel", "ssh")
    for cmd in commands:
        display(f"  {cmd}: {shutil.which(cmd) or 'not available in this shell'}")
    try:
        cap = probe_runtime(profile.runtime)
        display(f"Runtime: {cap.runtime_identity}; private key-file auth: {cap.auth_key_file}")
    except AppError as exc:
        display(f"Runtime: {exc.message} (run bash install.sh to install the managed runtime)")
    if root.exists():
        mode = root.stat().st_mode & 0o777
        display(f"State permissions: {oct(mode)}" + (" — should be 0700" if mode & 0o077 else " — private"))
    else:
        display("State directory will be created privately on first session.")
    if (root / "models").exists():
        for model in _library(root).list():
            display(f"Model: {model.name}; path {'present' if Path(model.path).is_file() else 'MISSING'}; context {model.supported_context or 'unverified'}")
    display("WEHI offload, step flags, terminals and compute-node web egress require the explicit smoke test.")
    return 0


def _wait_ready(manager, manifest, timeout=300):
    deadline = time.monotonic() + timeout
    delay, previous = 0.2, None
    while time.monotonic() < deadline:
        manifest = manager.load(manifest.id)
        if not manifest.demo:
            manifest = manager.status(manifest)
        if session_is_ended(manifest):
            raise AppError("scheduler", "This session has ended. Use Restore saved chats in the launcher to start a new allocation with its saved conversation.")
        if manifest.backend_state == "stopped" and manifest.started_at:
            raise AppError("startup", "The session service has stopped. Check allocation status in the launcher before restoring saved chats.")
        if manifest.endpoint and manifest.backend_state in {"ready", "generating", "awaiting_approval", "reloading"}:
            return manifest
        if manifest.backend_state == "failed":
            raise AppError("startup", manifest.error or "Backend failed to start; run doctor and inspect private session diagnostics.")
        status = f"{manifest.scheduler_state} / {manifest.backend_state}" + (f" · {manifest.scheduler_reason}" if manifest.scheduler_reason else "")
        if status != previous:
            display(f"Session {manifest.id[:8]}: {status}", flush=True)
            previous = status
        time.sleep(delay)
        delay = min(5, delay * 1.5)
    raise AppError("startup", "Session is still queued or loading. It remains saved; use ./hpc-llm attach to resume later.")


def _spawn_demo(manager, root):
    model = ModelSpec(id="demo", name="DEMO · synthetic streaming model", path="demo", supported_context=32768, thinking="enable_thinking", capability_provenance="Controlled fake backend fixture, not a real model")
    settings = InferenceSettings(web_provider="fixture")
    manifest = manager.create(model, ResourceRequest(), SiteProfile(name="demo", modules=[]), settings, demo=True)
    log_path = Path(manifest.directory) / "supervisor.log"
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "ab") as output:
        proc = subprocess.Popen([sys.executable, "-m", "hpc_llm.cli", "--state-root", str(root.resolve()), "_supervise", manifest.id], stdin=subprocess.DEVNULL, stdout=output, stderr=output, start_new_session=True)
    # The supervisor writes its own PID/endpoint; launcher never overwrites its manifest.
    return _wait_ready(manager, manifest, 30)


def attach(manager, manifest, *, return_to_launcher=True):
    from .lifecycle import inside_allocation
    if manifest.demo or (manifest.profile.scheduler == "slurm" and inside_allocation(manifest)):
        result = asyncio.run(run_tui(manifest))
    else:
        argv = manager.attach_argv(manifest)
        display("Attaching to your existing GPU session. Detach leaves it running.", flush=True)
        # Launcher fullscreen app has exited before the step owns this terminal.
        try:
            from .lifecycle import scheduler_environment
            from .terminal import run_attached
            result = run_attached(argv, env=scheduler_environment())
        except KeyboardInterrupt:
            display("\nDetached. The allocation remains active; use attach or stop.")
            return 130
    if result == 42 and return_to_launcher:
        return launcher(manager, manager.root)
    return result


async def run_tui(manifest):
    from .client import SessionClient
    from .ui import ChatApp
    from .lifecycle import inside_allocation
    if not manifest.demo and not inside_allocation(manifest):
        raise AppError("permission", "Chat must run inside its allocated compute job. Use ./hpc-llm attach.")
    client = SessionClient(manifest)
    try:
        await client.state()  # Verify identity and reachability before opening the TUI.
        result = await ChatApp(client).run_async()
    finally:
        await client.close()
    return 42 if isinstance(result, dict) and result.get("action") == "new_session" else 0


def launcher(manager, root, resume_only=False):
    from .ui import LauncherApp
    from .profiles import ProfileStore
    store = ProfileStore(root, APP_ROOT)
    error_message = ""
    while True:
        profile = profile_for(root)
        library = _library(root)
        inventory = manager.inventory(profile)
        sessions = []
        status_errors = []
        for session in manager.list():
            try:
                sessions.append(manager.status(session))
            except AppError as exc:
                status_errors.append(f"Session {session.id[:8]} unavailable: {exc.message}")
        if status_errors:
            error_message = "\n".join(filter(None, [error_message, *status_errors]))
        choice = LauncherApp(library.list(), sessions, profile, model_service=library, inventory=inventory, error_message=error_message, profile_service=store).run()
        error_message = ""
        if not choice:
            return 0
        action = choice.get("action")
        if action == "reload":
            continue
        if action == "resume":
            try:
                manifest = _wait_ready(manager, manager.load(choice["session_id"]))
                result = attach(manager, manifest, return_to_launcher=False)
                if result == 42:
                    continue
                if result not in (0, 130):
                    error_message = "Could not attach to that allocation. Check its refreshed status and try again."
                    continue
                return result
            except AppError as exc:
                error_message = exc.message
                continue
        if action == "register":
            library.register(choice["path"], **({"projector_path": choice["projector"]} if choice.get("projector") else {}))
            continue
        if action == "remote":
            # Exact filename/revision must be selected before this action returns.
            if choice.get("filename"):
                library.download(choice["repo_id"], choice["filename"], choice["revision"])
            else:
                display(json.dumps(library.list_remote(choice["repo_id"], choice.get("revision", "main")), indent=2, default=str))
                input("Press Enter to return to model selection.")
            continue
        if action == "restart":
            try:
                previous = manager.load(choice["session_id"])
                resources = ResourceRequest.model_validate(choice.get("resources", previous.resources.model_dump()))
                replacement = manager.restart_saved(previous, resources, profile=profile)
                replacement = manager.submit(replacement)
                result = attach(manager, _wait_ready(manager, replacement), return_to_launcher=False)
                if result == 42:
                    continue
                if result not in (0, 130):
                    error_message = "The restored session could not attach. Check its status under Resume before trying again."
                    continue
                return result
            except AppError as exc:
                error_message = exc.message
                continue
        if action == "start":
            model = next((m for m in library.list() if m.id == choice["model_id"]), None)
            if not model:
                raise AppError("validation", "Selected model is no longer registered.")
            resources = ResourceRequest.model_validate(choice.get("resources", profile.resources.model_dump()))
            settings = InferenceSettings.model_validate(choice.get("settings", model.defaults or {"context": 8192 if model.projector_path else 4096}))
            settings.threads = min(settings.threads, max(1, resources.cpus - 1))
            settings.threads_batch = min(settings.threads_batch, max(1, resources.cpus - 1))
            manifest = manager.create(model, resources, profile, settings)
            manifest = manager.submit(manifest)
            result = attach(manager, _wait_ready(manager, manifest), return_to_launcher=False)
            if result == 42:
                continue
            return result
        return 0


def smoke(args, manager):
    if not 1 <= args.minutes <= 30:
        raise AppError("validation", "Smoke walltime must be 1–30 minutes")
    path = Path(args.model).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise AppError("validation", "Select an existing complete GGUF model file; the smoke test never downloads weights.")
    from .models import validate_model
    metadata, _ = validate_model(path, args.projector)
    if args.mtp:
        from .models import validate_mtp
        validate_mtp(Path(args.mtp), metadata)
    profile = profile_for(args.state_root)
    if args.runtime:
        profile.runtime = args.runtime
    resource_data = profile.resources.model_dump()
    resource_data["walltime"] = f"00:{args.minutes:02d}:00"
    for field in ("memory_gb", "gpu_type"):
        if getattr(args, field) is not None:
            resource_data[field] = getattr(args, field)
    resources = ResourceRequest.model_validate(resource_data)
    from .contracts import SessionManifest
    from .lifecycle import scheduler_for
    preview_root = args.state_root.expanduser().resolve()
    preview = SessionManifest(owner=getpass.getuser(), directory=str(preview_root / "sessions" / "dry-run"),
                              model=ModelSpec(name=path.name, path=str(path)), profile=profile, resources=resources)
    batch_preview = scheduler_for(profile).render_batch(preview, APP_ROOT / "hpc-llm", preview_root)
    display(batch_preview)
    display(f"{profile.name} smoke ({profile.scheduler}): {resources.gpu_count} {resources.gpu_type} on {resources.partition}, {resources.cpus} CPUs, {resources.memory_gb} GB RAM, " + resources.walltime)
    display(f"Existing model: {path}")
    if args.projector:
        display(f"Vision projector: {Path(args.projector).expanduser().resolve()}")
    display(f"Acceleration: {args.acceleration}; MTP head: {args.mtp or 'embedded if present'} (runtime/model compatibility still requires loading)")
    display(f"Managed runtime: {profile.runtime}; no site modules loaded. Authenticated loopback ports chosen dynamically.")
    display("Checks still needed: real offload, attach, mid-response reconnect, thinking/context, files, optional web.")
    if not args.execute:
        display("DRY RUN — no allocation submitted. Add --execute only after reviewing resources and the model.")
        return 0
    model = _library(args.state_root).register(str(path), projector_path=args.projector, mtp_path=args.mtp)
    settings = InferenceSettings(thinking="auto", temperature=0, max_tokens=128, context=8192 if args.projector else 4096, acceleration=args.acceleration)
    manifest = manager.submit(manager.create(model, resources, profile, settings))
    display(f"Session: {manifest.id}\nRelease it with: ./hpc-llm stop {manifest.id}")
    return attach(manager, _wait_ready(manager, manifest))


def configure_terminal():
    """Select colours before Textual imports; preserve deliberate overrides."""
    if not sys.stdout.isatty() or os.environ.get("TEXTUAL_COLOR_SYSTEM") or os.environ.get("NO_COLOR"):
        return
    terminal = os.environ.get("TERM", "")
    advertised = os.environ.get("COLORTERM", "").lower()
    if advertised in {"truecolor", "24bit"}:
        os.environ["TEXTUAL_COLOR_SYSTEM"] = "truecolor"
    elif terminal.endswith("256color"):
        os.environ["TEXTUAL_COLOR_SYSTEM"] = "256"
    elif terminal in {"linux", "xterm-16color"}:
        # Observed OOD terminals advertise limited colours despite rendering
        # the explicitly selected 256-colour mode correctly.
        # Use the conservative 256 palette; real Linux consoles can override
        # with TEXTUAL_COLOR_SYSTEM=standard.
        os.environ["TEXTUAL_COLOR_SYSTEM"] = "256"


def main(argv=None):
    configure_terminal()
    args = parse_args(argv)
    root = args.state_root.expanduser().resolve()
    os.umask(0o077)
    try:
        if args.check:
            return check(root)
        if args.command == "doctor":
            return doctor(root)
        if args.command == "profiles":
            return profiles_command(args, root)
        manager = _manager(root)
        if args.command in {"_supervise", "_supervisor"}:
            from .supervisor import run_supervisor
            manifest = manager.load(args.session_id)
            from .lifecycle import current_job_id
            job_id = current_job_id(manifest.profile)
            if not manifest.demo and (not job_id or (manifest.job_id and job_id != manifest.job_id)):
                raise AppError("permission", "Supervisor must run in its allocated batch job.")
            return asyncio.run(run_supervisor(Path(manifest.directory) / "manifest.json")) or 0
        if args.command == "_tui":
            return asyncio.run(run_tui(manager.load(args.session_id)))
        if args.command == "sessions":
            for session in manager.list():
                status = manager.status(session) if not session.demo else session
                display(f"{status.id}  {'DEMO' if status.demo else status.scheduler_state}  {status.backend_state}  {status.model.name}")
            return 0
        if args.command == "stop":
            manifest = manager.load(args.session_id)
            if manifest.demo:
                async def stop_demo():
                    from .client import SessionClient
                    client = SessionClient(manifest)
                    try:
                        await client.request("POST", "/stop", json={})
                    finally:
                        await client.close()
                asyncio.run(stop_demo())
            else:
                manager.stop(manifest)
            display("Session stopped. Saved chats and exports are retained.")
            return 0
        if args.command == "attach":
            if args.session_id:
                manifest = manager.load(args.session_id)
                return attach(manager, _wait_ready(manager, manifest))
            return launcher(manager, root, resume_only=True)
        if args.command == "models":
            library = _library(root)
            if args.model_action == "register":
                result = library.register(args.path, **({"projector_path":args.projector} if args.projector else {}), **({"mtp_path":args.mtp} if args.mtp else {}))
            elif args.model_action == "remote":
                result = library.list_remote(args.repo, args.revision)
            elif args.model_action == "install":
                plan = library.plan_install(args.source, args.quant, args.revision, args.projector, mtp=args.mtp, model_id=args.model, force_verify=args.verify_full)
                display(json.dumps(plan, indent=2))
                if args.dry_run:
                    display("DRY RUN — metadata checked; no model weights downloaded.")
                    return 0
                result = library.execute_install(plan)
                display(f"Downloaded to: {result}" if isinstance(result, Path) else "Model registered. Start a new GPU session to load added companions; existing sessions retain their current model files.")
            elif args.model_action == "download":
                result = library.download(args.repo, args.filename, args.revision, projector_filename=args.projector_filename, mtp_filename=args.mtp_filename)
            else:
                result = library.list()
            display(json.dumps(result, default=lambda x: x.model_dump() if hasattr(x,"model_dump") else str(x), indent=2))
            return 0
        if args.command == "smoke":
            return smoke(args, manager)
        if args.demo:
            existing = [m for m in manager.list() if m.demo and m.backend_state not in {"failed", "stopped"}]
            if existing:
                # Verify a stale manifest instead of silently starting an extra process.
                try:
                    return attach(manager, existing[0])
                except AppError:
                    display("Previous demo endpoint is unavailable; starting a fresh DEMO session.")
            return attach(manager, _spawn_demo(manager, root))
        return launcher(manager, root)
    except KeyboardInterrupt:
        display("\nStopped waiting. Submitted allocations remain active; use sessions, attach or stop.")
        return 130
    except (AppError, ValueError, OSError) as exc:
        display(f"hpc-llm: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
