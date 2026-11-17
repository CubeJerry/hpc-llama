"""Compute-owned native llama.cpp UI with authenticated portable browser access."""
from __future__ import annotations
import asyncio
import hmac
import os
from pathlib import Path
import signal
import socket
import time
import zlib
from urllib.parse import unquote
import aiohttp
from aiohttp import web
from .contracts import AppError, InferenceSettings, ResourceRequest
from .lifecycle import atomic_json, read_private_json, private_token, current_job_id, walltime_seconds


def ssh_command(host, remote_port, backend_port):
    from .connections import port_number, ssh_base
    base = ssh_base(host)
    return base[:-1] + ['-N', '-R', f'127.0.0.1:{port_number(remote_port)}:127.0.0.1:{port_number(backend_port)}', base[-1]]


async def run_native(manifest):
    from .backend import Backend
    from .connections import ReverseTunnel, authenticated_probe, ood_prefix, validate_origin
    folder = Path(manifest.directory)
    config = read_private_json(folder / 'browser.json')
    backend = Backend(manifest)
    backend.native_webui = True
    options = config.get('connection', {'mode': 'ood', 'ood_origin': config.get('ood_origin', '')})
    mode = options['mode']
    if mode not in {'ood', 'local-forward', 'reverse-ssh'} or config.get('transport') not in {'compute', 'portable'}:
        raise AppError('startup', 'This session uses the retired login gateway. Stop it and start a browser session from the launcher.')
    backend.native_host = '127.0.0.1'
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    tunnel = None
    gateway = None
    listener = None
    manifest.supervisor_pid = os.getpid()
    manifest.started_at = time.time()
    manifest.expires_at = manifest.started_at + walltime_seconds(manifest.resources.walltime)
    manifest.node = socket.getfqdn()
    manifest.job_id = current_job_id(manifest.profile) or manifest.job_id
    manifest.scheduler_state = 'RUNNING'
    manifest.backend_state = 'loading'
    def save():
        atomic_json(folder / 'manifest.json', manifest.model_dump())
    save()
    (folder / 'browser-access.json').unlink(missing_ok=True)
    try:
        loading = asyncio.create_task(backend.start())
        stopping = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait((loading, stopping), return_when=asyncio.FIRST_COMPLETED)
            if stopping in done:
                return  # finally closes the backend even during model loading.
            manifest.capabilities = await loading
        finally:
            for task in (loading, stopping):
                if not task.done():
                    task.cancel()
            await asyncio.gather(loading, stopping, return_exceptions=True)
        async with aiohttp.ClientSession(headers={'Authorization': 'Bearer ' + backend.token}) as client:
            async with client.get(backend.endpoint + '/', timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200 or 'text/html' not in response.headers.get('Content-Type', ''):
                    raise AppError('startup', 'This llama.cpp runtime does not serve its native web UI. Install a runtime with the embedded UI enabled.')
        atomic_json(folder / 'browser-backend.json', {'token': backend.token, 'endpoint': backend.endpoint})
        listener = socket.socket()
        listener.bind(('0.0.0.0' if mode == 'ood' else '127.0.0.1', 0))
        listener.listen(128)
        listener.setblocking(False)
        public_port = listener.getsockname()[1]
        prefix = ood_prefix(options.get('ood_path', '/rnode/{node}/{port}'), manifest.node, public_port) if mode == 'ood' else ''
        origin = validate_origin(options.get('ood_origin', '')) if mode == 'ood' else None
        password = private_token(folder / 'browser.password')
        workspace = None
        if config.get('files') is not None:
            from .browser_files import BrowserFiles
            workspace = BrowserFiles(backend.endpoint, folder / 'browser-backend.json', config['files'])
        gateway = web.AppRunner(gateway_app(backend.endpoint, folder / 'browser-backend.json',
            password, prefix, origin, cookie_auth=True, workspace=workspace), access_log=None, shutdown_timeout=5)
        await gateway.setup()
        await web.SockSite(gateway, listener).start()
        await authenticated_probe(f'http://127.0.0.1:{public_port}', password, prefix=prefix, origin=origin)
        access = {'url': origin + prefix + '/' if origin else f'http://127.0.0.1:{options.get("local_port", 8765)}/',
                  'node': manifest.node, 'port': public_port, 'auth': 'cookie', 'mode': mode}
        if mode == 'reverse-ssh':
            tunnel = ReverseTunnel(options, public_port, folder / 'logs' / 'reverse-ssh.log')
            connecting = asyncio.create_task(tunnel.start(password))
            stopping = asyncio.create_task(stop.wait())
            try:
                done, _ = await asyncio.wait((connecting, stopping), return_when=asyncio.FIRST_COMPLETED)
                if stopping in done:
                    return
                access['relay_port'] = await connecting
            finally:
                for task in (connecting, stopping):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(connecting, stopping, return_exceptions=True)
        manifest.backend_state = 'ready'
        manifest.backend_pid = backend.pid
        manifest.endpoint = backend.endpoint
        save()
        atomic_json(folder / 'browser-access.json', access)
        while not stop.is_set():
            if tunnel is not None and tunnel.process.returncode is not None:
                raise AppError('network', 'Reverse SSH disconnected. Inspect logs/reverse-ssh.log; the GPU job is releasing its resources.')
            if backend.process.returncode is not None:
                raise AppError('startup', 'llama.cpp exited; inspect the backend log.')
            try:
                await asyncio.wait_for(stop.wait(), 1)
            except TimeoutError:
                pass
    except Exception as exc:
        manifest.backend_state = 'failed'
        manifest.error = str(exc)
        save()
        raise
    finally:
        (folder / 'browser-access.json').unlink(missing_ok=True)
        try:
            if tunnel is not None:
                await tunnel.close()
        finally:
            try:
                if gateway is not None:
                    await gateway.cleanup()
            finally:
                if listener is not None:
                    listener.close()
                await backend.close()
                (folder / 'browser-backend.json').unlink(missing_ok=True)
                manifest.endpoint = None
                manifest.backend_pid = None
                manifest.supervisor_pid = None
                if manifest.backend_state != 'failed':
                    manifest.backend_state = 'stopped'
                save()


HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'content-length', 'host',
               'authorization', 'cookie', 'origin', 'referer', 'accept-encoding'}


async def native_html(response):
    """Bound compressed input and decoded output from the embedded single-file UI."""
    compressed_limit, decoded_limit = 8 * 1024 * 1024, 16 * 1024 * 1024
    body = bytearray()
    async for chunk in response.content.iter_chunked(65536):
        body.extend(chunk)
        if len(body) > compressed_limit:
            raise AppError('startup', 'Compressed native UI exceeds the supported 8 MiB size.')
    encoding = response.headers.get('Content-Encoding', '').lower().strip()
    if encoding == 'gzip':
        try:
            decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
            decoded = decoder.decompress(body, decoded_limit + 1)
            if len(decoded) > decoded_limit or decoder.unconsumed_tail or not decoder.eof or decoder.unused_data:
                raise ValueError('Oversized or malformed gzip')
            return decoded
        except (ValueError, zlib.error):
            raise AppError('startup', 'Native UI gzip is invalid or exceeds the supported 16 MiB decoded size.') from None
    if encoding not in {'', 'identity'}:
        raise AppError('startup', 'Native UI uses an unsupported compression encoding.')
    return bytes(body)


def gateway_app(upstream, token_path, password, prefix='', public_origin=None, *, cookie_auth=False, workspace=None):
    """Authenticated streaming HTTP proxy. No arbitrary target URLs or file paths."""
    app = web.Application(client_max_size=256 * 1024 * 1024)
    from .browser_login import BrowserLogin
    login = BrowserLogin(password, prefix, bool(public_origin)) if cookie_auth else None
    async def lifetime(app):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
                                         auto_decompress=False) as client:
            app['client'] = client
            try:
                yield
            finally:
                if workspace is not None:
                    await workspace.close()
    app.cleanup_ctx.append(lifetime)
    async def proxy(request):
        origin = request.headers.get('Origin')
        expected_origin = public_origin.rstrip('/') if public_origin else f'{request.scheme}://{request.host}'
        if origin and origin != expected_origin:
            raise web.HTTPForbidden(text='Cross-origin requests are not accepted.')
        if request.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise web.HTTPForbidden(text='Cross-site requests are not accepted.')
        path = request.rel_url.raw_path
        if prefix and path.startswith(prefix + '/'):
            path = path[len(prefix):]
        elif prefix and path == prefix:
            path = '/'
        # rnode strips its prefix; node may retain it. Support both explicitly.
        if not path.startswith('/') or path.startswith('//'):
            raise web.HTTPBadRequest()
        # Canonicalise once and reject ambiguous encodings before local routing or
        # upstream forwarding. The native runtime's /tools shell executor is never
        # an application tool, even when the workspace panel is disabled.
        try:
            decoded = unquote(path, errors='strict')
        except UnicodeError:
            raise web.HTTPBadRequest(text='Invalid URL encoding.') from None
        if decoded.startswith('//') or '%' in decoded or '\\' in decoded or any(ord(c) < 32 for c in decoded) or any(p in {'.', '..'} for p in decoded.split('/')):
            raise web.HTTPBadRequest(text='Ambiguous URL path.')
        if decoded.rstrip('/') == '/tools' or decoded.startswith('/tools/'):
            raise web.HTTPForbidden(text='Native shell tools are disabled. Use the bounded Files & tools panel.')
        if decoded != path and decoded.startswith('/_hpc'):
            raise web.HTTPBadRequest(text='Encoded application routes are not accepted.')
        if login is not None:
            response = await login.gate(request, path)
            if response is not None:
                return response
            if path == '/_hpc/session':
                return web.json_response({'ok': True}, headers={'Cache-Control': 'no-store'})
        else:
            try:
                auth = aiohttp.BasicAuth.decode(request.headers.get('Authorization', ''))
                authorized = auth.login == 'hpc' and hmac.compare_digest(auth.password, password)
            except ValueError:
                authorized = False
            if not authorized:
                return web.Response(status=401, headers={'WWW-Authenticate': 'Basic realm="HPC llama.cpp", charset="UTF-8"'})
        if workspace is not None:
            local = await workspace.handle(request, decoded)
            if local is not None:
                return local
        result = None
        try:
            token = read_private_json(token_path)['token']
            headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS and not k.lower().startswith('x-forwarded-')}
            headers['Authorization'] = 'Bearer ' + token
            headers['Accept-Encoding'] = 'gzip, identity'
            url = upstream + path + ('?' + request.rel_url.raw_query_string if request.rel_url.raw_query_string else '')
            async with app['client'].request(request.method, url, headers=headers,
                    data=request.content.iter_chunked(65536), allow_redirects=False) as response:
                outgoing = {k: v for k, v in response.headers.items() if k.lower() not in HOP_HEADERS | {'set-cookie'}}
                outgoing.update({'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no',
                                 'Referrer-Policy': 'no-referrer', 'X-Content-Type-Options': 'nosniff'})
                if (workspace is not None or login is not None) and 'text/html' in response.headers.get('Content-Type', '') and response.status == 200:
                    body = await native_html(response)
                    outgoing = {key: value for key, value in outgoing.items() if key.lower() not in {'content-encoding', 'etag', 'content-md5', 'digest', 'accept-ranges', 'content-range'}}
                    if login is not None:
                        body = login.initialize_html(body)
                    if workspace is not None:
                        body = workspace.transform_html(body, prefix)
                    return web.Response(body=body, status=response.status, headers=outgoing)
                # Relative asset references keep the stock UI under the OOD prefix.
                result = web.StreamResponse(status=response.status, headers=outgoing)
                await result.prepare(request)
                async for chunk in response.content.iter_chunked(65536):
                    await result.write(chunk)
                return result
        except (AppError, aiohttp.ClientError, OSError):
            if result is not None and result.prepared:
                result.force_close()
                return result
            return web.Response(status=503, text='Model or reverse tunnel unavailable. Check ./hpc-llm sessions and the session logs; reload this page when ready.')
    app.router.add_route('*', '/{path:.*}', proxy)
    return app


def add_arguments(sub):
    parser = sub.add_parser('web', help='Start or reconnect to a compute-owned native browser chat')
    actions = parser.add_subparsers(dest='web_action', required=True)
    start = actions.add_parser('start', help='Submit one persistent GPU job using configured connection mode')
    start.add_argument('--model', required=True, help='Registered model ID, name, or full GGUF path')
    start.add_argument('--projector'); start.add_argument('--mtp')
    start.add_argument('--context', type=int)
    start.add_argument('--acceleration', choices=['off', 'auto', 'embedded', 'mtp'])
    start.add_argument('--connection', choices=['ood', 'local-forward', 'reverse-ssh'])
    start.add_argument('--ssh-host', help='Workstation SSH destination; {node} is the allocated compute hostname')
    start.add_argument('--jump-host', help='Workstation SSH jump host/alias')
    start.add_argument('--relay-host', help='Stable relay hostname/alias reachable FROM compute nodes')
    start.add_argument('--relay-jump-host', help='Compute-to-relay SSH jump host/alias')
    start.add_argument('--local-port', type=int, help='Browser port on your computer; no local process is started here')
    start.add_argument('--relay-port', type=int, help='Loopback relay port; zero allocates an available port')
    start.add_argument('--workspace', type=Path, help='Exact workspace under the selected profile allowed roots')
    start.add_argument('--preset', help='Named preset; CLI resource values override it for this job')
    for field in ('walltime', 'gpu-type', 'partition', 'account', 'qos', 'constraint'):
        start.add_argument('--' + field)
    for field in ('memory-gb', 'cpus', 'gpu-count'):
        start.add_argument('--' + field, type=int)
    connect = actions.add_parser('connect', help='Redisplay access to an existing job; never submits another allocation')
    connect.add_argument('session_id')
    start.add_argument('--ood-url', help='Your Open OnDemand origin, e.g. https://ondemand.example.edu')
    return parser


def command(args, manager, root):
    """All browser services live in the GPU job, including reconnects."""
    from .cli import _library, profile_for, display, _wait_ready
    from .compute_web import is_compute_browser, prepare_browser, show_browser
    from .contracts import ConnectionConfig
    from .config import resolve_resources
    if args.web_action == 'connect':
        manifest = manager.status(manager.load(args.session_id))
        if not is_compute_browser(manifest):
            raise AppError('startup', 'This is an older gateway session. Stop it and start a browser session from the launcher.')
        return show_browser(_wait_ready(manager, manifest))
    profile = profile_for(root)
    options = profile.connection.model_dump()
    for arg, field in [('connection', 'mode'), ('ssh_host', 'ssh_host'), ('jump_host', 'jump_host'),
                       ('relay_host', 'relay_host'), ('relay_jump_host', 'relay_jump_host'),
                       ('local_port', 'local_port'), ('relay_port', 'relay_port'), ('ood_url', 'ood_origin')]:
        value = getattr(args, arg, None)
        if value is not None:
            options[field] = value
    profile.connection = ConnectionConfig.model_validate(options)
    library = _library(root)
    models = library.list()
    model = next((model for model in models if args.model == model.id), None)
    if model is None:
        canonical = str(Path(args.model).expanduser().resolve())
        matches = [model for model in models if canonical == model.path or args.model == model.name]
        if len(matches) > 1:
            raise ValueError('Model name is ambiguous; use its registered ID or exact file path.')
        model = matches[0] if matches else None
    if model is None or args.projector or args.mtp:
        model = library.register(model.path if model else args.model,
            **({'projector_path': args.projector} if args.projector else {}),
            **({'mtp_path': args.mtp} if args.mtp else {}))
    overrides = {field: getattr(args, field) for field in ResourceRequest.model_fields if getattr(args, field, None) is not None}
    resources = resolve_resources(profile, preset=args.preset, cli=overrides)
    settings = InferenceSettings.model_validate(model.defaults or {})
    for field in ('context', 'acceleration'):
        value = getattr(args, field, None)
        if value is not None:
            setattr(settings, field, value)
    settings.threads = min(settings.threads, max(1, resources.cpus - 1))
    settings.threads_batch = min(settings.threads_batch, max(1, resources.cpus - 1))
    manifest = manager.create(model, resources, profile, settings)
    prepare_browser(manifest, root, args.ood_url, workspace=args.workspace)
    manifest = manager.submit(manifest)
    display(f'Submitted job {manifest.job_id}. Closing this launcher leaves the allocation running.\nReconnect: ./hpc-llm attach {manifest.id}', flush=True)
    return show_browser(_wait_ready(manager, manifest))
