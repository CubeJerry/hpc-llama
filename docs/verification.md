# Verification and cluster smoke tests

Local checks use deterministic scheduler/model fixtures and real loopback HTTP. They exercise request handling and real file writes, rather than real model inference or an external OOD installation.

## Run the local checks

Use the installed application's Python, or a development virtual environment with this project's dependencies. The login JavaScript check also requires Node.js. From the repository root:

```bash
(
  for check in check_config check_launcher_config check_connections check_browser_login check_browser check_compute_web check_browser_files check_mtp_install check_mtp_browser check_file_download check_file_refresh; do
    PYTHONPATH="$PWD/src" .venv/bin/python "scripts/$check.py" || exit 1
  done
)
```

| Check | Boundary covered |
|---|---|
| `check_config.py` | YAML validation, migration/backups, resource precedence, arbitrary scheduler labels and unsupported resource requests. |
| `check_launcher_config.py` | CLI/state precedence, TUI resource overrides and protection of saved presets. |
| `check_connections.py` | Connection plans, compute-owned service lifecycle, readiness, cleanup, SSH options, terminal links and control-character rejection. |
| `check_browser_login.py` | Fragment-login JavaScript and cookie authentication through an Authorization-stripping, path/cookie-rewriting proxy. |
| `check_browser.py` | HTTP streaming, bodies, prefix handling, origin protection, unavailable backends and legacy gateway compatibility. |
| `check_compute_web.py` | Compute-owned OOD service, queue/loading states, ready link publication, shutdown and launcher behavior with scheduler/model fixtures. |
| `check_browser_files.py` | Browser-to-tool API, UTF-8/CSV writes, approvals, changed-since-preview rejection, directories/downloads, filesystem escapes/secrets, bounded model execution/results and cancellation. |
| `check_mtp_install.py`, `check_mtp_browser.py` | Companion installation and capability-gated runtime arguments. |
| `check_file_download.py`, `check_file_refresh.py` | Existing file download/refresh behavior. |

For the optional Chromium interaction, provide the development-only `playwright` package and its installed Chromium browser, or a system `chromium` executable, in the test environment:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python scripts/check_browser_files.py --browser
```

This test is designed to open the private fragment link through the stripping/rewrite proxy and exercise the injected panel's actual saves, overwrite denial, download, refresh and model task. Its native chat is an HTML fixture. During this release's verification, Chromium startup was blocked by the environment denying `AF_UNIX socket()` with `EPERM`; the real browser path therefore remains unverified.

For the actual OpenSSH transport fixture, a usable `sshd` is required:

```bash
PYTHONPATH="$PWD/src" .venv/bin/python scripts/check_connections.py --real-ssh
```

That fixture creates temporary test keys/configuration and a local test daemon; it does not edit the user's SSH files. This release's environment also denied the SSH Unix control socket with `EPERM`, so the real OpenSSH round trip could not complete. Connection-plan/lifecycle fixtures do not establish working site forwarding.

Independent HTTP checks exercised a backend fixture through the authenticated gateway and a proxy that strips Authorization and rewrites paths/cookies: gzip HTML integration, assets, model discovery, a base64-image completion body, SSE, unauthorised API rejection and real approved `.txt`/`.csv` writes. This establishes protocol handling with deterministic fixtures. The shipped b11351 UI, image inference, GPU startup and each site's external route need the cluster checks below.

## Check your cluster

Use a small model already registered on the cluster, an existing shared workspace, and a short allocation. First check the installation without allocating:

```bash
./hpc-llm --check
./hpc-llm doctor
./hpc-llm models list
```

Launch through the TUI with your configured route, or submit an OOD session directly:

```bash
./hpc-llm web start --model REGISTERED_MODEL_ID \
  --connection ood --ood-url https://YOUR-OOD-HOST \
  --workspace /absolute/shared/workspace --context 4096 --walltime 00:20:00
```

For WEHI, copy the HTTPS origin from your working Milton OOD page, without its path. For local forwarding, replace the connection flags with `--connection local-forward --ssh-host '{node}' --jump-host YOUR-LOGIN-ALIAS`. For reverse forwarding, use `--connection reverse-ssh --relay-host FIXED-RELAY`. Run the printed local-forward command on your computer. Provision approved SSH authentication and known hosts first.

Confirm the following with the real route and runtime:

1. The queue/loading state is shown, and the ready link appears after the model API is ready. Interrupt waiting, then use `attach SESSION_ID`; it must show the same job without allocating again.
2. Open the complete private URL. Confirm that the key disappears from the address bar, no Basic Auth popup appears, and no native API-key entry is needed.
3. In the browser network panel, check assets, model discovery and streamed completion requests under the configured URL prefix. Send a prompt and verify incremental output.
4. With a compatible vision model/projector, attach a small PNG or JPEG and verify an image response. This checks the real upload payload and inference, not only the HTML page.
5. Open **Files & tools**. Read a workspace text file, create a directory, preview/save `smoke.txt` and `smoke.csv`, then download and inspect both. Reject an overwrite and confirm the old file remains. Try a model file task and inspect its approval and executed result.
6. Close the launcher and browser; check that the scheduler job remains active. Reconnect, confirm origin-dependent browser history, and export a conversation. For SSH modes, close/restart the workstation forward; it must not create or cancel a GPU job.
7. Stop the session and verify that the scheduler allocation ends with `./hpc-llm stop SESSION_ID`, then `./hpc-llm sessions`.

For reverse SSH, also test a relay outage during a short disposable session: the service should mark failure, clean up its reverse tunnel and release the allocation. For each scheduler/site, verify partition permissions, account/QoS handling, walltime and the requested host RAM/CPU/GPU resources in the scheduler's job report.

Record the runtime build, route, node, browser and scheduler result when reporting a site issue. Never include the private link, session keys, cookies or backend credential.
