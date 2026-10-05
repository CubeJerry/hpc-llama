"""Read-only runtime discovery and capability-gated llama.cpp command construction."""
from __future__ import annotations
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
from .contracts import AppError, BackendCapabilities, InferenceSettings, ModelSpec, ResourceRequest


def sanitized_environment(threads: int = 1) -> dict[str, str]:
    # Retain module-set library/CUDA paths and scheduler GPU visibility, never server overrides.
    env = {k: v for k, v in os.environ.items() if not k.startswith(('LLAMA_', 'GGML_', 'OMP_', 'MKL_', 'OPENBLAS_', 'VECLIB_'))}
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = str(threads)
    return env


def _probe(command: list[str], argument: str) -> str:
    try:
        result = subprocess.run(command + [argument], capture_output=True, text=True, timeout=8,
                                env=sanitized_environment(), check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ''
    return (result.stdout + '\n' + result.stderr)[:262144] if result.returncode == 0 else ''


def probe_runtime(override: str = '') -> BackendCapabilities:
    candidates = []
    if override:
        # A whole existing path with spaces is one executable; other input is argv, never a shell.
        try:
            pieces = [override] if Path(override).is_file() else shlex.split(override)
        except ValueError as exc:
            raise AppError('validation', 'Runtime selection contains unmatched quotes. Choose the executable path.') from exc
        if not pieces:
            raise AppError('validation', 'Choose a llama runtime executable.')
        resolved = shutil.which(pieces[0])
        if not resolved:
            raise AppError('startup', 'The selected runtime executable was not found.')
        base = [str(Path(resolved).resolve())] + pieces[1:]
        if len(base) == 1 and Path(resolved).name == 'llama':
            base.append('serve')
        candidates.append(base)
    else:
        app_root = Path(os.environ.get('HPC_LLM_ROOT', Path(__file__).resolve().parents[2]))
        managed = app_root / 'runtime' / 'bin' / 'llama-server'
        if managed.is_file() and os.access(managed, os.X_OK):
            candidates.append([str(managed)])
    for command in candidates:
        help_text = _probe(command, '--help')
        flags = sorted(set(re.findall(r'(?<!\w)--[a-z][a-z0-9-]*', help_text)))
        required = {'--host', '--port', '--model', '--ctx-size', '--parallel', '--alias', '--api-key-file'}
        if not required <= set(flags):
            continue
        version = _probe(command, '--version').strip().splitlines()
        controls = [name for name, flag in RUNTIME_FLAGS.items() if flag in flags]
        # Values belong to the --spec-type option, not arbitrary help prose.
        spec_lines = []
        collecting = False
        for line in help_text.splitlines():
            if line.lstrip().startswith('-'):
                collecting = '--spec-type' in line.split()
            if collecting:
                spec_lines.append(line)
        mtp_supported = '--spec-type' in flags and bool(re.search(
            r'(?<![\w-])draft-mtp(?![\w-])', '\n'.join(spec_lines)))
        if mtp_supported:
            controls.append('acceleration')
            controls.extend(name for name, flag in MTP_FLAGS.items() if flag in flags)
        flash_help = next((line for line in help_text.splitlines() if '--flash-attn' in line), '')
        if 'flash_attention' in controls and not all(word in flash_help for word in ('on', 'off', 'auto')):
            controls.remove('flash_attention')
        return BackendCapabilities(runtime_identity=(version[0][:200] if version else 'Unreported build'), command=command,
            flags=flags, auth_key_file=True, runtime_controls=controls, mtp_supported=mtp_supported,
            sampling=[key for key, flag in SAMPLING_FLAGS.items() if flag in flags],
            provenance='Installed executable --help and --version; template capabilities pending authenticated /props')
    raise AppError('unsupported', 'No compatible managed llama.cpp server found. Run bash install.sh to install the private runtime, or explicitly select another executable. Required: --api-key-file, --alias, --parallel, --ctx-size, --host and --port. No model was loaded.')


RUNTIME_FLAGS = {'gpu_layers': '--gpu-layers', 'threads': '--threads', 'threads_batch': '--threads-batch',
    'batch_size': '--batch-size', 'ubatch_size': '--ubatch-size', 'cache_type_k': '--cache-type-k',
    'cache_type_v': '--cache-type-v', 'flash_attention': '--flash-attn'}
MTP_FLAGS = {'spec_draft_n_max': '--spec-draft-n-max', 'spec_draft_n_min': '--spec-draft-n-min',
    'spec_draft_p_min': '--spec-draft-p-min'}
DRAFT_MODEL_FLAGS = ('--spec-draft-model', '--model-draft')


def acceleration_command(capabilities, model, settings) -> list[str]:
    """Resolve optional MTP conservatively; startup readiness is verified separately."""
    capabilities.acceleration_status = 'Off'
    capabilities.acceleration_reason = 'MTP is disabled'
    if settings.acceleration == 'off':
        return []
    reason = ''
    if not capabilities.mtp_supported or '--spec-type' not in capabilities.flags:
        reason = 'Installed runtime does not advertise draft-mtp. Install a compatible runtime.'
    elif not model.mtp_path and not model.mtp_layers:
        reason = 'No embedded MTP layers or registered MTP companion. Add a matching companion in Models.'
    draft_flag = next((flag for flag in DRAFT_MODEL_FLAGS if flag in capabilities.flags), None)
    if model.mtp_path and not draft_flag:
        reason = 'Installed runtime cannot select a separate MTP companion.'
    if reason:
        capabilities.acceleration_status = 'Unavailable'
        capabilities.acceleration_reason = reason
        if settings.acceleration == 'mtp':
            raise AppError('unsupported', reason)
        return []
    command = ['--spec-type', 'draft-mtp']
    if model.mtp_path:
        command += [draft_flag, str(Path(model.mtp_path).expanduser().resolve())]
    defaults = InferenceSettings()
    for name, flag in MTP_FLAGS.items():
        if flag in capabilities.flags:
            command += [flag, str(getattr(settings, name))]
        elif getattr(settings, name) != getattr(defaults, name):
            raise AppError('unsupported', f'The installed runtime does not support {name}.')
    capabilities.acceleration_status = 'Requested'
    capabilities.acceleration_reason = 'MTP arguments configured; waiting for backend readiness'
    return command


SAMPLING_FLAGS = {'temperature': '--temp', 'top_p': '--top-p', 'top_k': '--top-k', 'min_p': '--min-p',
    'seed': '--seed', 'repeat_penalty': '--repeat-penalty', 'presence_penalty': '--presence-penalty',
    'frequency_penalty': '--frequency-penalty'}


def owned_gpu_memory(pid: int) -> str:
    """Optional compute-side corroboration; never return information about other PIDs."""
    executable = shutil.which('nvidia-smi')
    if not executable:
        return 'GPU process memory unverified (nvidia-smi unavailable)'
    try:
        result = subprocess.run([executable, '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=3, check=False)
        if result.returncode == 0:
            total = 0
            found = False
            for row in result.stdout[:65536].splitlines():
                columns = [column.strip() for column in row.split(',')]
                if len(columns) == 2 and columns[0] == str(pid) and columns[1].isdigit():
                    total += int(columns[1])
                    found = True
            if found:
                return f'Owned backend GPU process memory: {total} MiB'
    except (OSError, subprocess.TimeoutExpired):
        pass
    return 'GPU process memory unverified (owned PID not reported)'


def allocated_gpu_memory(pid: int) -> dict[str, int]:
    """Measure only a single visible GPU identified by the owned backend PID.

    Numeric CUDA indices may be remapped by Slurm/container CUDA device order,
    so the PID's reported UUID, not a host index guess, selects the device.
    MIG, multiple visible devices and missing visibility remain unknown.
    """
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    devices = [item.strip() for item in visible.split(',') if item.strip()]
    if len(devices) != 1 or not (devices[0].isdigit() or devices[0].startswith('GPU-')):
        return {}
    executable = shutil.which('nvidia-smi')
    if not executable:
        return {}
    deadline = time.monotonic() + 3

    def query(arguments):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        result = subprocess.run([executable, *arguments, '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            timeout=remaining, check=False)
        if result.returncode or len(result.stdout) > 65536:
            return []
        return [[value.strip() for value in row.split(',')]
                for row in result.stdout.splitlines() if row.strip()]

    try:
        rows = query(['--query-compute-apps=pid,gpu_uuid,used_memory'])
        owned = [row for row in rows if len(row) == 3 and row[0] == str(pid)]
        if len(owned) != 1 or not owned[0][1].startswith('GPU-') or not owned[0][2].isdigit():
            return {}
        uuid, used = owned[0][1], int(owned[0][2])
        if devices[0].startswith('GPU-') and not uuid.startswith(devices[0]):
            return {}
        rows = query(['--id=' + uuid, '--query-gpu=uuid,memory.total,memory.free,mig.mode.current'])
        if len(rows) != 1 or len(rows[0]) != 4:
            return {}
        gpu, total, free, mig = rows[0]
        if gpu != uuid or not total.isdigit() or not free.isdigit() or mig not in ('Disabled', '[N/A]', 'N/A'):
            return {}
        total, free = int(total), int(free)
        if not 0 < used <= total or not 0 <= free <= total or used + free > total:
            return {}
        return {'gpu_memory_total_bytes': total * 2**20,
                'gpu_memory_free_bytes': free * 2**20,
                'gpu_memory_used_bytes': used * 2**20}
    except (OSError, subprocess.TimeoutExpired, TimeoutError, ValueError):
        return {}


def build_backend_command(capabilities: BackendCapabilities, model: ModelSpec, settings: InferenceSettings,
                          resources: ResourceRequest, port: int, key_file: Path | str) -> list[str]:
    if not capabilities.auth_key_file:
        raise AppError('auth', 'Runtime lacks private API-key-file authentication; refusing to start.')
    if not 1024 <= port <= 65535:
        raise AppError('validation', 'Invalid local inference port.')
    if max(settings.threads, settings.threads_batch) > max(1, resources.cpus - 1):
        raise AppError('validation', f'Use at most {max(1, resources.cpus - 1)} inference threads; one allocation CPU is reserved for the TUI and services.')
    if model.supported_context and settings.context > model.supported_context:
        raise AppError('validation', f'Requested context exceeds the model metadata limit ({model.supported_context}).')
    # key_file parent is the unique private session directory; alias verifies endpoint identity.
    command = list(capabilities.command) + ['--model', str(Path(model.path).resolve()), '--host', '127.0.0.1',
        '--port', str(port), '--api-key-file', str(Path(key_file).resolve()), '--alias', 'hpc-' + Path(key_file).parent.name,
        '--parallel', '1', '--ctx-size', str(settings.context)]
    defaults = InferenceSettings()
    for name, flag in RUNTIME_FLAGS.items():
        value = getattr(settings, name)
        supported = flag in capabilities.flags and (name != 'flash_attention' or name in capabilities.runtime_controls)
        if supported:
            command += [flag, str(value)]
        elif value != getattr(defaults, name):
            raise AppError('unsupported', f'The installed runtime does not support {name}.')
    for flag in ('--jinja', '--no-webui', '--no-slots', '--offline', '--no-context-shift'):
        if flag in capabilities.flags:
            command.append(flag)
    if '--threads-http' in capabilities.flags:
        command += ['--threads-http', '1']
    if '--log-verbosity' in capabilities.flags:
        command += ['--log-verbosity', '3']
    if model.projector_path:
        if '--mmproj' not in capabilities.flags:
            raise AppError('unsupported', 'This runtime does not advertise multimodal projector support.')
        command += ['--mmproj', str(Path(model.projector_path).resolve())]
    command += acceleration_command(capabilities, model, settings)
    return command
