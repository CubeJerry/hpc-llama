"""Owned persistent inference subprocess and authenticated streaming client."""
from __future__ import annotations
import asyncio
import contextlib
import json
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import sys
import time
import aiohttp
from aiohttp.http_exceptions import LineTooLong
from .contracts import AppError, BackendCapabilities, InferenceSettings, MAX_DOCUMENT_ARGUMENT_CHARS, estimate_prompt_tokens, response_token_budget, inspect_reasoning_template
from .runtime import probe_runtime, build_backend_command, sanitized_environment, owned_gpu_memory, allocated_gpu_memory, RUNTIME_FLAGS, SAMPLING_FLAGS, acceleration_command

STARTUP_TIMEOUT = 600
DOCUMENT_TOOL = 'write_workspace_file'
NORMAL_TOOL_CHARS = 65536
MAX_INFERENCE_EVENT_BYTES = MAX_DOCUMENT_ARGUMENT_CHARS + 65536


def _append_tool_fragment(call, fragment):
    """Bound every append, including arguments arriving before a fragmented name."""
    function = fragment.get('function', {})
    if not isinstance(function, dict):
        raise AppError('backend', 'Malformed native tool function.')
    additions = {'id': fragment.get('id', ''), 'name': function.get('name', ''),
                 'arguments': function.get('arguments', '')}
    if any(not isinstance(value, str) for value in additions.values()):
        raise AppError('backend', 'Malformed native tool fragment.')
    # Tool metadata must not borrow the document-content allowance.
    if any(len(call[key]) + len(additions[key]) > 4096 for key in ('id', 'name')):
        raise AppError('backend', 'Native tool metadata exceeded the size limit.')
    name = call['name'] + additions['name']
    arguments_size = len(call['arguments']) + len(additions['arguments'])
    # A missing/partial name may still resolve to the document tool. This staging
    # allowance is bounded and is revoked immediately on divergence or completion.
    document_candidate = DOCUMENT_TOOL.startswith(name)
    limit = MAX_DOCUMENT_ARGUMENT_CHARS if document_candidate else NORMAL_TOOL_CHARS
    metadata_size = 0 if document_candidate else len(call['id']) + len(additions['id']) + len(name)
    if arguments_size + metadata_size > limit:
        raise AppError('backend', 'Tool arguments exceeded the bounded size limit.')
    for key, addition in additions.items():
        call[key] += addition


def _validate_tool_size(call):
    size = len(call['arguments']) if call['name'] == DOCUMENT_TOOL else sum(map(len, call.values()))
    limit = MAX_DOCUMENT_ARGUMENT_CHARS if call['name'] == DOCUMENT_TOOL else NORMAL_TOOL_CHARS
    if size > limit:
        raise AppError('backend', 'Tool arguments exceeded the bounded size limit.')


def private_file(path: Path, text: str):
    """Create credentials exclusively; never follow a pre-existing link."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as handle:
            handle.write(text)
    except OSError as exc:
        raise AppError('permission', 'Cannot safely create private backend credential.') from exc


def available_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class Backend:
    def __init__(self, manifest):
        self.manifest = manifest
        self.settings = manifest.settings.model_copy(deep=True)
        self.capabilities = manifest.capabilities.model_copy(deep=True)
        self.process = None
        self.endpoint = ''
        self.token = ''
        self._client = None
        self._log_task = None
        self._memory_task = None
        self._key_file = None
        self._lock = asyncio.Lock()
        self._streaming = False
        self._preflight_cache = None
        self._gpu_kv_buffers = {}
        self.diagnostics = {'offload': 'Not yet verified', 'own_process_memory': 'Unverified', 'oom': False, 'port_collision': False}

    @property
    def pid(self):
        return self.process.pid if self.process and self.process.returncode is None else None

    async def _read_logs(self):
        # Raw logs can contain prompts; retain ONLY parsed technical facts.
        while self.process and self.process.stdout:
            raw = await self.process.stdout.readline()
            if not raw:
                break
            line = raw.decode(errors='replace')[:8192]
            match = re.search(r'offloaded\s+(\d+)/(\d+)\s+layers', line, re.I)
            if match:
                self.diagnostics['offload'] = f'{match[1]}/{match[2]} layers offloaded'
            kv = re.search(r'CUDA(\d+)\s+KV buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB', line)
            if kv:
                device = int(kv[1])
                self._gpu_kv_buffers[device] = self._gpu_kv_buffers.get(device, 0) + int(float(kv[2]) * 2**20)
            if re.search(r'out of memory|cuda.*alloc.*fail|CUDA error.*memory', line, re.I):
                self.diagnostics['oom'] = True
            if re.search(r'address already in use|failed to bind|cannot bind', line, re.I):
                self.diagnostics['port_collision'] = True

    @staticmethod
    def _file_identities(paths):
        identities = []
        for path in paths:
            info = Path(path).stat()
            identities.append((str(Path(path).resolve()), info.st_dev, info.st_ino,
                               info.st_size, info.st_mtime_ns, info.st_ctime_ns))
        return tuple(identities)

    async def _collect_memory(self, process):
        # Optional diagnostics must not hold up authenticated readiness. Cancelling
        # to_thread cannot stop nvidia-smi, so publish only for this live process.
        if self.settings.gpu_layers == 0:
            return
        results = await asyncio.gather(
            asyncio.to_thread(owned_gpu_memory, process.pid),
            asyncio.to_thread(allocated_gpu_memory, process.pid), return_exceptions=True)
        memory = results[0] if isinstance(results[0], str) else 'GPU memory diagnostic unavailable'
        measurements = results[1] if isinstance(results[1], dict) else {}
        if self.process is process and process.returncode is None:
            self.diagnostics['own_process_memory'] = memory
            self.capabilities.provenance += '; ' + memory
            for name, value in measurements.items():
                setattr(self.capabilities, name, value)
            if measurements and len(self._gpu_kv_buffers) == 1:
                self.capabilities.gpu_kv_bytes = next(iter(self._gpu_kv_buffers.values()))

    def _preflight(self, settings=None):
        """Reuse discovery on reload only while the runtime and model files match."""
        settings = settings or self.settings
        mtp_path = self.manifest.model.mtp_path if settings.acceleration != 'off' else None
        selection = (self.manifest.profile.runtime, self.manifest.model.path,
                     self.manifest.model.projector_path, mtp_path)
        if self._preflight_cache:
            cached_selection, paths, identities, capabilities = self._preflight_cache
            try:
                if selection == cached_selection and self._file_identities(paths) == identities:
                    return capabilities.model_copy(deep=True)
            except OSError:
                pass  # Normal discovery produces the actionable missing-file error.
        capabilities = probe_runtime(self.manifest.profile.runtime)
        from .models import validate_model
        metadata, model_paths = validate_model(Path(self.manifest.model.path), self.manifest.model.projector_path, mtp_path)
        self.manifest.model.memory_metadata = {k: v for k, v in metadata.items() if k == 'general.architecture' or k.endswith(('.block_count', '.embedding_length', '.attention.head_count', '.attention.head_count_kv', '.attention.key_length', '.attention.value_length', '.attention.sliding_window'))}
        layers = metadata.get(str(metadata.get('general.architecture', '')) + '.nextn_predict_layers', 0)
        self.manifest.model.mtp_layers = layers if isinstance(layers, int) and not isinstance(layers, bool) and layers > 0 else 0
        paths = list(model_paths)
        if mtp_path:
            paths.append(mtp_path)
        if self.manifest.model.projector_path:
            paths.append(self.manifest.model.projector_path)
        paths.extend(arg for arg in capabilities.command if Path(arg).is_file())
        self._preflight_cache = (selection, paths, self._file_identities(paths), capabilities.model_copy(deep=True))
        return capabilities

    async def start(self, settings=None):
        async with self._lock:
            return await self._start(settings or self.settings)

    async def _start(self, settings):
        if self.pid:
            raise AppError('busy', 'The inference server is already running.')
        settings = InferenceSettings.model_validate(settings.model_dump())
        if max(settings.threads, settings.threads_batch) > max(1, self.manifest.resources.cpus - 1):
            raise AppError('validation', 'Inference threads must leave one allocation CPU for the TUI and services.')
        directory = Path(self.manifest.directory)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise AppError('permission', 'Session directory must be owned by you and private (mode 700).')
        if self.manifest.demo:
            capabilities = BackendCapabilities(runtime_identity='DEMO fake-http-v1', auth_key_file=True,
                thinking='enable_thinking', native_tools=True, loaded_context=settings.context,
                runtime_controls=list(RUNTIME_FLAGS), provenance='Deterministic fixture, not a real model or CUDA validation')
            acceleration_command(capabilities, self.manifest.model, settings)
        else:
            capabilities = await asyncio.to_thread(self._preflight, settings)
        last_error = None
        for attempt in range(3):
            capabilities.memory_settings = settings.model_dump()
            self._gpu_kv_buffers = {}
            for field in ('gpu_memory_total_bytes', 'gpu_memory_free_bytes', 'gpu_memory_used_bytes', 'gpu_kv_bytes'):
                setattr(capabilities, field, None)
            self.diagnostics.update(oom=False, port_collision=False, own_process_memory='Unverified')
            port = available_port()
            self.endpoint = f'http://127.0.0.1:{port}'
            self.token = secrets.token_urlsafe(32)
            self._key_file = directory / ('backend-' + secrets.token_hex(8) + '.key')
            private_file(self._key_file, self.token + '\n')
            alias = 'hpc-' + directory.name
            if self.manifest.demo:
                command = [sys.executable, '-m', 'hpc_llm.fake_backend', '--port', str(port), '--key-file', str(self._key_file),
                           '--alias', alias, '--context', str(settings.context)]
            else:
                try:
                    command = build_backend_command(capabilities, self.manifest.model, settings, self.manifest.resources, port, self._key_file)
                except BaseException:
                    await self._close()
                    raise
            try:
                self.process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT, env=sanitized_environment(settings.threads), limit=2**20)
                self._log_task = asyncio.create_task(self._read_logs())
                self._client = aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=None, connect=3, sock_read=120),
                    headers={'Authorization': 'Bearer ' + self.token}, read_bufsize=(MAX_INFERENCE_EVENT_BYTES + 1) // 2)
                self.capabilities = capabilities
                deadline = time.monotonic() + (15 if self.manifest.demo else STARTUP_TIMEOUT)
                while time.monotonic() < deadline:
                    if self.process.returncode is not None:
                        if self.diagnostics['oom']:
                            raise AppError('startup', 'Model loading ran out of memory. Reduce context or use a smaller quantization; request a larger GPU if necessary.')
                        raise AppError('startup', 'Inference process exited during startup. Inspect doctor and runtime/module compatibility; no prompt was retried.')
                    try:
                        async with self._client.get(self.endpoint + '/v1/models', timeout=aiohttp.ClientTimeout(total=2)) as response:
                            if response.status == 401:
                                raise AppError('auth', 'Backend authentication failed; refusing the endpoint.')
                            if response.status == 200:
                                listing = await response.json()
                                if alias not in [m.get('id') for m in listing.get('data', [])]:
                                    raise AppError('auth', 'Local port belongs to a different model/session.')
                                await self._verify_properties(settings)
                                # Explicitly prove authentication is enforced, not only accepted.
                                async with self._client.get(self.endpoint + '/v1/models', headers={'Authorization': 'Bearer invalid-probe'}, timeout=aiohttp.ClientTimeout(total=2)) as denied:
                                    if denied.status not in (401, 403):
                                        raise AppError('auth', 'Backend did not enforce API authentication; refusing to expose it.')
                                if self.process.returncode is not None:
                                    raise AppError('startup', 'Inference process died during readiness check.')
                                if not self.manifest.demo:
                                    self.diagnostics['own_process_memory'] = 'GPU memory diagnostic pending'
                                    self._memory_task = asyncio.create_task(self._collect_memory(self.process))
                                if self.capabilities.acceleration_status == 'Requested':
                                    self.capabilities.acceleration_status = 'Enabled'
                                    self.capabilities.acceleration_reason = ('Backend ready with MTP arguments; draft acceptance and speed are unverified')
                                self.settings = settings.model_copy(deep=True)
                                self.manifest.backend_pid = self.pid
                                self.manifest.capabilities = self.capabilities
                                return self.capabilities
                    except (aiohttp.ClientError, TimeoutError):
                        pass
                    await asyncio.sleep(0.1)
                raise AppError('startup', 'Model did not become ready within the bounded startup timeout. Check model size and runtime compatibility.')
            except BaseException as exc:
                collision = self.diagnostics['port_collision']
                await self._close()
                last_error = exc
                if isinstance(exc, AppError) and (exc.code == 'auth' or collision) and attempt < 2:
                    continue
                raise
        raise last_error or AppError('startup', 'Could not allocate a private backend port.')

    async def _verify_properties(self, settings):
        async with self._client.get(self.endpoint + '/props', timeout=aiohttp.ClientTimeout(total=2)) as response:
            if response.status != 200:
                raise AppError('unsupported', 'Authenticated /props is required to verify actual context and template support.')
            props = await response.json()
        loaded = props.get('default_generation_settings', {}).get('n_ctx')
        if not isinstance(loaded, int) or loaded < 512 or props.get('total_slots') != 1:
            raise AppError('unsupported', 'Cannot verify the single-slot effective context from this runtime.')
        self.capabilities.loaded_context = loaded
        self.capabilities.supported_context = self.manifest.model.supported_context
        template = props.get('chat_template', '')
        self.capabilities.thinking = 'enable_thinking' if 'enable_thinking' in template else 'unknown'
        self.capabilities.reasoning_efforts, self.capabilities.reasoning_default = inspect_reasoning_template(template)
        template_caps = props.get('chat_template_caps', {})
        self.capabilities.native_tools = template_caps.get('supports_tool_calls') is True
        self.capabilities.vision = bool(self.manifest.model.projector_path) and props.get('modalities', {}).get('vision') is True
        if not self.manifest.demo:
            self.capabilities.provenance = 'Installed help; authenticated /props template, modalities and single-slot n_ctx; ' + self.diagnostics['offload']
        self._thinking_kwargs(settings.thinking)

    def _thinking_kwargs(self, thinking):
        if thinking == 'auto':
            return {}
        if thinking in ('on', 'off'):
            if self.capabilities.thinking != 'enable_thinking':
                raise AppError('unsupported', 'Thinking On/Off is not verified for this template. Use Auto.')
            return {'enable_thinking': thinking == 'on'}
        if thinking not in self.capabilities.reasoning_efforts:
            raise AppError('unsupported', f'This template does not advertise reasoning effort {thinking}. Use Auto or an available level.')
        kwargs = {'reasoning_effort': thinking}
        if self.capabilities.thinking == 'enable_thinking':
            kwargs['enable_thinking'] = True
        return kwargs

    def request_payload(self, messages, settings, tools=None):
        thinking_kwargs = self._thinking_kwargs(settings.thinking)
        defaults = InferenceSettings()
        payload = {'model': 'hpc-' + Path(self.manifest.directory).name, 'messages': messages,
                   'stream': True, 'max_tokens': response_token_budget(
                       settings.max_tokens, self.capabilities.loaded_context, estimate_prompt_tokens(messages, tools))}
        for name in SAMPLING_FLAGS:
            if name in self.capabilities.sampling:
                payload[name] = getattr(settings, name)
            elif getattr(settings, name) != getattr(defaults, name):
                raise AppError('unsupported', f'This installed runtime does not advertise {name}.')
        if thinking_kwargs:
            payload['chat_template_kwargs'] = thinking_kwargs
        if tools:
            if not self.capabilities.native_tools:
                raise AppError('unsupported', 'Native tool calling is not verified. Use application-managed file/web actions.')
            payload['tools'] = tools
            payload['tool_choice'] = 'auto'
        return payload

    async def stream(self, messages, settings, tools=None):
        if not self.pid or not self._client:
            raise AppError('backend', 'The inference server is not running. Start a new session after checking diagnostics.')
        if self._streaming:
            raise AppError('busy', 'Another reply is already using this model.')
        payload = self.request_payload(messages, settings, tools)
        self._streaming = True
        calls = {}
        finished = False
        try:
            async with self._client.post(self.endpoint + '/v1/chat/completions', json=payload) as response:
                if response.status != 200:
                    raw = (await response.content.read(8192)).decode(errors='replace')
                    if 'context' in raw.lower():
                        raise AppError('context_overflow', 'Prompt plus response budget exceeds loaded context. Shorten history/files or reload with a larger context.')
                    if 'memory' in raw.lower():
                        raise AppError('backend', 'Inference ran out of memory. Reduce context or choose a smaller model.')
                    raise AppError('backend', f'Inference request failed (HTTP {response.status}); no automatic retry.')
                async for line in response.content:
                    if len(line) > MAX_INFERENCE_EVENT_BYTES:
                        raise AppError('backend', 'Inference event exceeded the size limit.')
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if data == b'[DONE]':
                        break
                    try:
                        event = json.loads(data)
                        choices = event.get('choices') or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = choice.get('delta') or {}
                        for field, kind in (('content', 'delta'), ('reasoning_content', 'reasoning')):
                            if isinstance(delta.get(field), str) and delta[field]:
                                yield {'kind': kind, 'text': delta[field]}
                        for fragment in delta.get('tool_calls') or []:
                            index = fragment.get('index', 0)
                            if not isinstance(index, int) or not 0 <= index < settings.tool_limit:
                                raise AppError('backend', 'Tool-call count exceeded the configured limit.')
                            call = calls.setdefault(index, {'id': '', 'name': '', 'arguments': ''})
                            _append_tool_fragment(call, fragment)
                        if choice.get('finish_reason'):
                            ids = set()
                            for call in calls.values():
                                _validate_tool_size(call)
                                if not call['id'] or call['id'] in ids or not call['name']:
                                    raise AppError('backend', 'Malformed or duplicate native tool-call ID.')
                                ids.add(call['id'])
                                try:
                                    if not isinstance(json.loads(call['arguments']), dict):
                                        raise ValueError
                                except (ValueError, TypeError) as exc:
                                    raise AppError('validation', 'Model generated invalid JSON tool arguments; nothing was executed.') from exc
                                yield {'kind': 'tool_call', **call}
                            yield {'kind': 'done', 'finish_reason': choice['finish_reason']}
                            finished = True
                            break
                    except (ValueError, TypeError, KeyError) as exc:
                        raise AppError('backend', 'Malformed inference stream; the partial answer is retained.') from exc
            if not finished:
                raise AppError('backend', 'Inference connection ended before a completion marker. The partial answer is retained.')
        except (ValueError, LineTooLong) as exc:
            # aiohttp also bounds a line while receiving it, before yielding it.
            raise AppError('backend', 'Malformed or oversized inference event; the partial answer is retained.') from exc
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise AppError('backend', 'Inference connection failed; no prompt was replayed. Check the session status.') from exc
        finally:
            self._streaming = False

    async def reload(self, settings):
        async with self._lock:
            if self._streaming:
                raise AppError('busy', 'Wait for or cancel the active response before reloading.')
            previous = self.settings.model_copy(deep=True)
            await self._close()
            try:
                return await self._start(settings)
            except Exception as requested_error:
                try:
                    await self._start(previous)
                except Exception as rollback_error:
                    raise AppError('startup', 'Reload failed and the previous model could not restart. Stop this session and check runtime diagnostics.') from rollback_error
                raise AppError('startup', 'Requested reload failed; previous settings and model have been restored.') from requested_error

    async def _close(self):
        if self._memory_task:
            self._memory_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._memory_task
            self._memory_task = None
        if self._client:
            await self._client.close()
            self._client = None
        if self.process and self.process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    self.process.kill()
                await self.process.wait()
        if self._log_task:
            self._log_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._log_task
            self._log_task = None
        if self._key_file:
            with contextlib.suppress(FileNotFoundError):
                self._key_file.unlink()
            self._key_file = None
        self.process = None
        self.manifest.backend_pid = None

    async def close(self):
        async with self._lock:
            await self._close()
