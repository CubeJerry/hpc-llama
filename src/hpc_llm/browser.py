"""Native llama.cpp UI: compute-owned OOD service and legacy gateway modes."""
from __future__ import annotations
import asyncio
import hmac
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import time
from urllib.parse import urlsplit
import aiohttp
from aiohttp import web
from .contracts import AppError, InferenceSettings, ResourceRequest
from .lifecycle import atomic_json, read_private_json, private_token, current_job_id, walltime_seconds


def ssh_command(host, remote_port, backend_port):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', host):
        raise ValueError('SSH host must be a hostname or an SSH config alias.')
    if any(not 1024 <= int(p) <= 65535 for p in (remote_port, backend_port)):
        raise ValueError('Tunnel ports must be between 1024 and 65535.')
    return ['ssh', '-N', '-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=15',
            '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
            '-R', f'127.0.0.1:{remote_port}:127.0.0.1:{backend_port}', host]


async def run_native(manifest):
    from .backend import Backend
    folder = Path(manifest.directory)
    config = read_private_json(folder / 'browser.json')
    backend = Backend(manifest)
    backend.native_webui = True
    compute = config.get("transport") == "compute"
    direct = config.get("transport") == "direct"
    backend.native_host = "0.0.0.0" if direct else "127.0.0.1"
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
    try:
        if not (direct or compute):
            # Check batch SSH before loading GPU weights. Never prompt in a job.
            probe = await asyncio.create_subprocess_exec('ssh', '-T', '-o', 'BatchMode=yes',
                '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15', config['ssh_host'], 'true',
                stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            try:
                code = await asyncio.wait_for(probe.wait(), 20)
            except TimeoutError:
                probe.kill()
                await probe.wait()
                raise AppError('network', 'Compute-to-login SSH timed out.')
            if code:
                raise AppError('network', 'Compute-to-login batch SSH failed. Configure an approved SSH key and known_hosts for this login host before starting.')
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
        port = urlsplit(backend.endpoint).port
        endpoint = backend.endpoint if compute else (f'http://{manifest.node}:{port}' if direct else f'http://127.0.0.1:{config["tunnel_port"]}')
        atomic_json(folder / 'browser-backend.json', {'token': backend.token, 'endpoint': endpoint})
        if not (direct or compute):
            with (folder / 'logs' / 'reverse-ssh.log').open('ab') as log:
                tunnel = await asyncio.create_subprocess_exec(*ssh_command(config['ssh_host'], config['tunnel_port'], port),
                    stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log)
        if compute:
            from .compute_web import validate_origin
            origin = validate_origin(config['ood_origin'])
            # Bind once to an OS-selected port; no fixed-port collision or SSH.
            listener = socket.socket()
            listener.bind(('0.0.0.0', 0))
            listener.listen(128)
            listener.setblocking(False)
            public_port = listener.getsockname()[1]
            prefix = f'/rnode/{manifest.node}/{public_port}'
            password = private_token(folder / 'browser.password')
            gateway = web.AppRunner(gateway_app(backend.endpoint, folder / 'browser-backend.json',
                                               password, prefix, origin), access_log=None, shutdown_timeout=5)
            await gateway.setup()
            await web.SockSite(gateway, listener).start()
            atomic_json(folder / 'browser-access.json', {'url': origin + prefix + '/',
                        'node': manifest.node, 'port': public_port})
        manifest.backend_state = 'ready'
        manifest.backend_pid = backend.pid
        manifest.endpoint = backend.endpoint
        save()
        while not stop.is_set():
            if tunnel is not None and tunnel.returncode is not None:
                raise AppError('network', 'Reverse SSH ended. Inspect logs/reverse-ssh.log; check SSH keys, known_hosts and forwarding permissions.')
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
        if gateway is not None:
            await gateway.cleanup()
        if listener is not None:
            listener.close()
        if tunnel and tunnel.returncode is None:
            tunnel.terminate()
            try:
                await asyncio.wait_for(tunnel.wait(), 5)
            except TimeoutError:
                tunnel.kill()
                await tunnel.wait()
        await backend.close()
        (folder / 'browser-backend.json').unlink(missing_ok=True)
        if manifest.backend_state != 'failed':
            manifest.backend_state = 'stopped'
        save()


HOP_HEADERS = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
               'te', 'trailer', 'transfer-encoding', 'upgrade', 'content-length', 'host',
               'authorization', 'cookie', 'origin', 'referer', 'accept-encoding'}


def gateway_app(upstream, token_path, password, prefix='', public_origin=None):
    """Authenticated streaming HTTP proxy. No arbitrary target URLs or file paths."""
    app = web.Application(client_max_size=256 * 1024 * 1024)
    async def lifetime(app):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
                                         auto_decompress=False) as client:
            app['client'] = client
            yield
    app.cleanup_ctx.append(lifetime)
    async def proxy(request):
        try:
            auth = aiohttp.BasicAuth.decode(request.headers.get('Authorization', ''))
            authorized = auth.login == 'hpc' and hmac.compare_digest(auth.password, password)
        except ValueError:
            authorized = False
        if not authorized:
            return web.Response(status=401, headers={'WWW-Authenticate': 'Basic realm="HPC llama.cpp", charset="UTF-8"'})
        origin = request.headers.get('Origin')
        expected_origin = public_origin.rstrip('/') if public_origin else f'{request.scheme}://{request.host}'
        if origin and origin != expected_origin:
            raise web.HTTPForbidden(text='Cross-origin requests are not accepted.')
        if request.headers.get('Sec-Fetch-Site') == 'cross-site':
            raise web.HTTPForbidden(text='Cross-site requests are not accepted.')
        path = request.rel_url.raw_path
        if prefix and path.startswith(prefix + '/'):
            path = path[len(prefix):]
        # rnode strips its prefix; node may retain it. Support both explicitly.
        if not path.startswith('/') or path.startswith('//'):
            raise web.HTTPBadRequest()
        result = None
        try:
            token = read_private_json(token_path)['token']
            headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS and not k.lower().startswith('x-forwarded-')}
            headers['Authorization'] = 'Bearer ' + token
            url = upstream + path + ('?' + request.rel_url.raw_query_string if request.rel_url.raw_query_string else '')
            async with app['client'].request(request.method, url, headers=headers,
                    data=request.content.iter_chunked(65536), allow_redirects=False) as response:
                outgoing = {k: v for k, v in response.headers.items() if k.lower() not in HOP_HEADERS | {'set-cookie'}}
                outgoing.update({'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no',
                                 'Referrer-Policy': 'no-referrer', 'X-Content-Type-Options': 'nosniff'})
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
    parser = sub.add_parser('web', help='Native llama.cpp browser interface; no TUI')
    actions = parser.add_subparsers(dest='web_action', required=True)
    serve = actions.add_parser('serve', help='Browser dashboard: models, sessions, files and native chat')
    serve.add_argument('--workspace', type=Path, help='Directory to browse and edit; defaults to the installation workspace folder')
    serve.add_argument('--transport', choices=['direct', 'reverse-ssh'], default='direct', help='Compute connection: direct matches the ComfyUI cluster setup')
    serve.add_argument('--ssh-host', help='Reverse SSH destination reachable from compute nodes')
    start = actions.add_parser('start', help='Submit a new GPU job and open its browser gateway')
    start.add_argument('--model', required=True, help='Registered model ID, name, or full GGUF path')
    start.add_argument('--projector'); start.add_argument('--mtp')
    start.add_argument('--context', type=int)
    start.add_argument('--acceleration', choices=['off', 'auto', 'embedded', 'mtp'])
    start.add_argument('--ssh-host', help='Login-node hostname or SSH alias reachable FROM compute nodes')
    for field in ('walltime', 'gpu-type', 'partition', 'account'):
        start.add_argument('--' + field)
    for field in ('memory-gb', 'cpus'):
        start.add_argument('--' + field, type=int)
    connect = actions.add_parser('connect', help='Reopen a gateway for an existing web session')
    connect.add_argument('session_id')
    for p in (start, connect, serve):
        p.add_argument('--port', type=int, default=8765, help='Login gateway port')
        p.add_argument('--ood-url', help='Your Open OnDemand origin, e.g. https://ondemand.example.edu')
    return parser


def command(args, manager, root):
    from .cli import _library, profile_for, display
    from .backend import available_port
    if args.web_action == 'serve':
        return serve_dashboard(args, manager, root)
    if args.web_action == 'connect':
        from .compute_web import is_compute_browser, show_browser
        from .cli import _wait_ready
        existing = manager.load(args.session_id)
        if is_compute_browser(existing):
            return show_browser(_wait_ready(manager, existing))
    if not 1024 <= args.port <= 65535:
        raise ValueError('Gateway port must be between 1024 and 65535.')
    if args.ood_url:
        parsed = urlsplit(args.ood_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
            raise ValueError('--ood-url must be an HTTPS origin without a path.')
    import shutil
    if not shutil.which('ssh'):
        raise AppError('startup', 'OpenSSH client (ssh) is required on the login and compute nodes.')
    # Fail before submitting if the gateway port is already occupied.
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(('0.0.0.0' if args.ood_url else '127.0.0.1', args.port))
    listener.listen(128)
    listener.setblocking(False)
    try:
        host = socket.getfqdn()
        if args.web_action == 'start':
            ssh_host = args.ssh_host or host
            tunnel_port = available_port()
            ssh_command(ssh_host, tunnel_port, args.port)  # Validate before submit.
            profile = profile_for(root)
            library = _library(root)
            models = library.list()
            model = next((m for m in models if args.model == m.id), None)
            if model is None:
                canonical = str(Path(args.model).expanduser().resolve())
                matches = [m for m in models if canonical == m.path or args.model == m.name]
                if len(matches) > 1:
                    raise ValueError('Model name is ambiguous; use its registered ID or exact file path.')
                model = matches[0] if matches else None
            if model is None or args.projector or args.mtp:
                model = library.register(model.path if model else args.model,
                    **({'projector_path': args.projector} if args.projector else {}),
                    **({'mtp_path': args.mtp} if args.mtp else {}))
            values = profile.resources.model_dump()
            for field in ('walltime', 'gpu_type', 'partition', 'account', 'memory_gb', 'cpus'):
                if getattr(args, field) is not None:
                    values[field] = getattr(args, field)
            resources = ResourceRequest.model_validate(values)
            settings = InferenceSettings.model_validate(model.defaults or {})
            for field in ('context', 'acceleration'):
                if getattr(args, field) is not None:
                    setattr(settings, field, getattr(args, field))
            settings.threads = min(settings.threads, max(1, resources.cpus - 1))
            settings.threads_batch = min(settings.threads_batch, max(1, resources.cpus - 1))
            manifest = manager.create(model, resources, profile, settings)
            config = {'ssh_host': ssh_host, 'gateway_host': host, 'tunnel_port': tunnel_port}
            atomic_json(Path(manifest.directory) / 'browser.json', config)
            manifest = manager.submit(manifest)
        else:
            manifest = manager.status(manager.load(args.session_id))
            from .contracts import session_is_ended
            if session_is_ended(manifest) or manifest.backend_state in ('failed', 'stopped'):
                raise ValueError('This session has ended or failed. Start a new web session.')
            config = read_private_json(Path(manifest.directory) / 'browser.json')
            if config.get('transport') == 'direct':
                raise ValueError('This is a dashboard session. Run ./hpc-llm web serve and open it from Sessions.')
            if config['gateway_host'] != host:
                raise ValueError(f"Reconnect on the original login host: {config['gateway_host']}")
        folder = Path(manifest.directory)
        password = private_token(folder / 'browser.password')
        prefix = f'/rnode/{host}/{args.port}' if args.ood_url else ''
        url = args.ood_url.rstrip('/') + prefix + '/' if args.ood_url else f'http://localhost:{args.port}/'
        display(f'Session: {manifest.id}\nJob: {manifest.job_id}\nOpen in a separate browser tab: {url}\nBrowser username: hpc\nBrowser password: {password}', flush=True)
        if not args.ood_url:
            display('On your computer, keep this SSH tunnel open:\n' + shlex.join(['ssh', '-N', '-L', f'{args.port}:127.0.0.1:{args.port}', f'{manifest.owner}@{host}']))
        display(f'Keep this gateway command running. Ctrl+C closes the gateway only.\nReconnect: ./hpc-llm web connect {manifest.id} --port {args.port}' + (f' --ood-url {args.ood_url}' if args.ood_url else ''))
        display(f'Stop GPU allocation: ./hpc-llm stop {manifest.id}\nThe page may show unavailable while the job queues and the model loads.', flush=True)
        web.run_app(gateway_app(f"http://127.0.0.1:{config['tunnel_port']}", folder / 'browser-backend.json', password, prefix, args.ood_url),
                    sock=listener, print=None, access_log=None)
        return 0
    finally:
        listener.close()


def serve_dashboard(args, manager, root):
    from .cli import APP_ROOT, display
    from .dashboard import dashboard_app
    from .lifecycle import private_dir
    if not 1024 <= args.port <= 65535:
        raise ValueError('Gateway port must be between 1024 and 65535.')
    if args.ood_url:
        parsed = urlsplit(args.ood_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ('', '/'):
            raise ValueError('--ood-url must be an HTTPS origin without a path.')
    host = socket.getfqdn()
    prefix = f'/rnode/{host}/{args.port}' if args.ood_url else ''
    workspace = args.workspace or APP_ROOT / 'workspace'
    workspace.mkdir(parents=True, exist_ok=True)
    password = private_token(private_dir(Path(root) / 'dashboard') / 'password')
    app = dashboard_app(manager, root, password, prefix=prefix, public_origin=args.ood_url,
                        workspace=workspace, ssh_host=args.ssh_host, transport=args.transport)
    listener = socket.socket()
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('0.0.0.0' if args.ood_url else '127.0.0.1', args.port))
        listener.listen(128)
        listener.setblocking(False)
        url = args.ood_url.rstrip('/') + prefix + '/' if args.ood_url else f'http://localhost:{args.port}/'
        display(f'Open the dashboard: {url}\nUsername: hpc\nPassword: {password}\nWorkspace: {workspace}\nKeep this command running. Ctrl+C closes the dashboard; GPU jobs remain active.', flush=True)
        if not args.ood_url:
            import getpass
            display('On your computer run:\n' + shlex.join(['ssh', '-N', '-L', f'{args.port}:127.0.0.1:{args.port}', f'{getpass.getuser()}@{host}']))
        web.run_app(app, sock=listener, print=None, access_log=None)
    finally:
        listener.close()
    return 0
