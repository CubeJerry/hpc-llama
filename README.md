# HPC Llama

Run GGUF models in a persistent Slurm GPU job. PBS Pro / OpenPBS is also supported. Use the terminal launcher on your cluster login node to install or register models, choose resources, and start a job. Open its **Open chat** link to chat in llama.cpp's native browser interface. The job keeps running after you close the launcher or browser.

The browser service runs inside the GPU job. There is no separate launch dashboard or CPU allocation. The original terminal-chat version is preserved on [`tui`](https://github.com/CubeJerry/hpc-llama/tree/tui) at commit [`c99048d`](https://github.com/CubeJerry/hpc-llama/commit/c99048d30e8ccb1c18f6d455bba9644f5ce380c2).

## Install or update

Run these commands on a **Linux x86_64 login node**. Keep the application folder on retained home or project storage at a path shared with compute nodes.

```bash
git clone https://github.com/CubeJerry/hpc-llama.git
cd hpc-llama
bash install.sh --profile slurm
./hpc-llm --check
```

`slurm` is the default for new installations. Set your cluster's partition, GPU resources and connection route in **Site setup…**. The optional `--profile wehi` example supplies WEHI Milton presets; PBS users can choose `--profile pbs`. The previous `generic` profile remains available for existing configurations.

The installer supplies private Python, application libraries, pinned llama.cpp build b11351, and CUDA user-space libraries. It needs Bash, `curl`, `tar`, `sha256sum`, network access for downloads, and scheduler clients. CUDA requires glibc 2.28 or later and a compatible NVIDIA driver on compute nodes. It does not install the kernel driver or download model weights. Keep the installation at the same path; rerun the installer after moving it. Use `--backend cpu` for the CPU runtime. `./hpc-llm --demo` runs the simulated legacy terminal backend without a scheduler or model.

To update a Git installation, finish or stop active sessions first, then run from its existing directory:

```bash
git pull --ff-only
bash install.sh
./hpc-llm --check
./hpc-llm
```

For an existing ZIP installation named `hpc-llm`, upload `hpc-llm-0.2.0-source.zip` beside that directory, then:

```bash
unzip -o hpc-llm-0.2.0-source.zip
cd hpc-llm
sha256sum -c SOURCE_CHECKSUMS.sha256
bash install.sh
./hpc-llm --check
./hpc-llm
```

The source ZIP excludes state, credentials, environments and model weights. Updating preserves those existing directories and your selected profile. Supply `--profile slurm` when updating to switch profiles explicitly. Do not replace your saved configuration with an example during an update. Already-running jobs retain their loaded code; restart them to use this release.

## Configure a cluster

The versioned configuration is `state/config.yaml` by default. The installer creates it, or migrates existing JSON settings on first use. Open **Site setup…** to edit the scheduler, partition or queue, GPU label, account, defaults and named resource presets. Routine launch changes do not require editing YAML.

This is a minimal Slurm configuration using SSH through a cluster login host:

```yaml
schema_version: 1
selected_profile: slurm
profiles:
  slurm:
    scheduler: slurm
    resources:
      partition: gpu
      gpu_type: ""
      gpu_count: 1
      cpus: 4
      memory_gb: 32
      walltime: "01:00:00"
    model_cache: cache/models
    workspaces:
      - /absolute/path/to/your/workspace
    connection:
      mode: local-forward
      ssh_host: "{node}"
      jump_host: my-cluster-login
      local_port: 8765
```

Replace the partition, GPU label, paths and SSH login alias with your site's values. Choose OOD or reverse SSH if your site requires a different connection route. Create your workspace on shared cluster storage. Keep it separate from application state and credentials. Partition and GPU names accept your cluster's labels. This release supports **one GPU on one node**; other GPU counts are rejected before submission. `memory_gb` requests host RAM, not GPU VRAM.

See [configuration and precedence](docs/configuration.md), [Slurm example](examples/config.slurm.yaml), and [WEHI example](examples/config.wehi.yaml) for presets, paths, runtime environment, tool limits and connection options. Keep passwords, private keys and tokens out of YAML. Select another config with `./hpc-llm --config /absolute/path/config.yaml`.

On WEHI Milton, select the `wehi` profile, set the partition and GPU resources available to your account, and copy the HTTPS origin from your working OOD page, without its path. The working route is the compute-node reverse proxy. It does not need compute-to-login SSH. Site access and partition permissions still apply.

## Install or register a model

Open **Models…**, paste a Hugging Face repository or model URL, choose a quantization, and review the download plan. The launcher shows download and verification progress. The CLI offers the same installation path:

```bash
./hpc-llm models install unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL --dry-run
./hpc-llm models install unsloth/Qwen3.8-27B-GGUF --quant UD-Q4_K_XL
```

Use your chosen repository and quantization. The installer selects one main quantization and an unambiguous matching vision projector. Specify exact companion filenames when several match. Downloads are pinned and verified; unchanged, previously verified files reuse their checksum result. Add `--verify-full` to read every byte again.

Register existing files at their exact paths without copying them:

```bash
./hpc-llm models register /absolute/path/model.gguf \
  --projector /absolute/path/mmproj.gguf \
  --mtp /absolute/path/mtp.gguf
./hpc-llm models list
```

Omit companions you do not have. To add downloaded companions to an existing model, select that model as the update target in **Models…**, or run:

```bash
./hpc-llm models install --model /absolute/path/model.gguf \
  --projector auto --mtp auto
```

For a manually registered model, supply its original `OWNER/REPOSITORY` as the source too. Exact filenames replace `auto` when selection is ambiguous. The main model must match that repository. A companion-only update retains the main weights, model identity and saved defaults. Start a new GPU session after adding companions.

Set context size and **MTP acceleration** before launch. **Auto** uses MTP when the model and runtime support it. **Embedded MTP** selects a model's embedded head. **Registered MTP sidecar** uses the registered companion and reports missing requirements. A downloaded or registered file is not proof that the runtime can load it. MTP can require more VRAM and does not guarantee faster generation.

For the requested Qwen3.8 Q4_K_P main model, BF16 projector and FastMTP sidecar bundle:

```bash
bash scripts/install-qwen38-q4kp.sh --dry-run
bash scripts/install-qwen38-q4kp.sh
```

The pinned bundle totals about 19.76 GB. Its separate reduced-vocabulary FastMTP sidecar requires the [publisher's patched runtime](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF#run-hauhaucs-fastmtp). Installing it does not patch the managed runtime. Auto uses the main model's verified embedded MTP head when supported.

## Launch, reconnect and stop

```bash
./hpc-llm
```

1. Select a registered model.
2. Select a preset or edit **Resources**: partition/queue, GPU type, CPUs, host RAM, account and time limit. Choose context and acceleration.
3. Choose the connection mode and select an existing shared workspace. The default application workspace is created automatically.
4. Review the effective request, then select **Start session**.
5. Wait for the queue and model startup. Open **Open chat** when the launcher reports readiness.

The ready message includes a terminal hyperlink and the complete URL as a fallback. Click support depends on the terminal, including the site's OOD terminal. Copy the full URL if the label is not clickable. Set `HPC_LLM_HYPERLINKS=off` to disable hyperlinks or `on` to request them explicitly. The app does not try to launch a browser on the remote node.

The private link ends in `#access=...`. Opening it exchanges that key for an HttpOnly session cookie, then removes the fragment from the address bar. Keep the complete link private. No Basic Auth popup is required; leave the native UI API-key setting empty. The backend API key stays private inside the job. If the terminal truncates the link, the sign-in page accepts the access key as a fallback.

Closing the launcher, pressing Ctrl+C while waiting, or closing the browser leaves the submitted job running. Show status or redisplay an existing job's link with:

```bash
./hpc-llm sessions
./hpc-llm attach SESSION_ID
```

**Resume** in the launcher reconnects too. Reconnection does not submit another allocation. If a job has ended, select the model and start a new one. Legacy **Restore saved chats** applies to terminal-chat snapshots, not native browser history.

Release the allocation explicitly when finished:

```bash
./hpc-llm stop SESSION_ID
```

You can also select **Stop** in the launcher. GPU time continues until you stop the job or its walltime expires. This release does not renew allocations or change their expiry.

## Choose browser access

| Mode | Browser route | Prerequisites | Process to keep running |
|---|---|---|---|
| `ood` | Browser → site's OOD reverse proxy → authenticated service in the GPU job | Approved compute host/port route and HTTPS OOD origin | GPU job; no personal SSH tunnel |
| `local-forward` | Browser → local SSH forward → compute-node loopback service | Your computer can SSH to the compute node, directly or through an approved jump host | Printed SSH command on your computer, plus GPU job |
| `reverse-ssh` | Browser → local SSH forward → relay loopback → compute-owned reverse SSH forward → GPU service | Compute node can batch-SSH to a fixed relay; your computer can SSH to that same relay | Printed SSH command on your computer; the job owns the reverse tunnel |

Open OnDemand reverse proxying and SSH reverse forwarding are separate mechanisms. Use OOD or direct local forwarding when compute nodes cannot SSH back to a relay. The OOD mode binds the authenticated entry point for the site's proxy route; the inference backend stays on loopback. SSH modes keep service and tunnel endpoints on loopback.

For SSH modes, run the printed `ssh -N -L ...` command **on your computer**, then open the printed localhost URL there. Closing that SSH process disconnects browser access but does not stop the job; rerun it to reconnect. Specify a stable host or SSH alias, with jump hosts where needed. A load-balanced login alias may lead to a different relay. Nondefault SSH ports belong in your own SSH configuration.

OpenSSH must support forwarding with normal host-key verification and noninteractive authentication. Provision known hosts and authentication using your site's approved procedure. The app does not create SSH keys, modify SSH configuration, enable `GatewayPorts`, disable host-key checking or forward your SSH agent. See [connection configuration](docs/configuration.md#connection-settings).

## Tools, workspaces and file creation

Open **Files & tools** in native chat to browse and refresh allowed cluster workspaces, read UTF-8 text/code/CSV/JSON, create directories, save generated text or code at an exact path, and download files. Choose a workspace before saving. You can create `.txt`, `.csv`, Markdown, JSON and code files; saving a CSV writes text and does not calculate spreadsheet formulas.

Paste a native chat response into the panel's editor to save it, or use **Ask the model to work with files** for a task with file tools. That task has its own context; it does not automatically see native chat or its attachments. The server executes the loaded model's validated tool calls, returns results to the model, and repeats within the configured limits. The panel displays activity and failures, and can export its task transcript as JSON.

Writes show the exact destination and proposed content/diff for approval, including overwrites. A file is reported as saved only after the write completes. By default every write needs approval. Setting `tools.allow_writes: true` deliberately permits model-driven writes inside the selected workspace without individual approval. Manual editor saves still show a preview. The executor rejects paths outside allowed workspaces, symlink escapes and application secrets. It has no shell-execution tool.

Native attachments are separate from workspace files. Uploading an attachment to chat does not save it at a named cluster path. Use the files panel to persist an output, and **Download** to copy a saved file to your computer. See [storage and browser history](docs/storage.md) for the distinction and export instructions.

The tool workflow has file-size, directory-entry, call-count and time limits. Model-driven operations need compatible tool-call support. The visible file controls also work without relying on a model to request a tool. Old terminal web-search tools are not automatically connected to native chat. See [the browser integration](docs/browser-integration.md) for the execution path and runtime-update checks.

## History and storage

Weights and downloaded companions live in your model cache. Registered local files stay where you registered them. Session settings and logs live under `STATE/sessions/SESSION_ID/`. Explicitly approved outputs live in their selected cluster workspace.

Native browser history and attachments live in the browser's IndexedDB, not in the cluster session directory. On the pinned UI, changing only an OOD `/rnode/NODE/PORT/` path under the same browser origin keeps the same history database visible. Changing hostname, scheme or browser-facing port creates a different origin; using another browser profile also changes storage. Clearing site data can remove history.

Export important chats through the native sidebar. A single conversation exports as JSONL; multiple conversations export as a ZIP. Browser downloads go to your computer's download destination. Existing conversations are not silently migrated or deleted. See [storage details and pinned source references](docs/storage.md).

## Troubleshooting

| Symptom | Check |
|---|---|
| OOD 404 | Confirm the configured origin, proxy path and compute hostname are permitted by the site. Local readiness cannot prove that the external OOD route is enabled. |
| 401 or a sign-in page | Reopen the complete private link, including `#access=...`, in the same tab. Keep native UI API-key settings empty. Use `attach` to redisplay the link. |
| 403 | Open the configured URL origin rather than another hostname or port. Check the proxy's forwarded path and cookie handling; cross-origin requests are rejected. |
| HTML loads but assets, models or streaming fail | Preserve the complete URL prefix and trailing slash. Check browser network errors for missing assets and `/v1/models` or completion requests. A successful HTML page alone does not verify the proxy. |
| Queue or startup delay | Run `sessions`, then `attach SESSION_ID`. Inspect scheduler reasons and private session logs. Waiting cancellation does not cancel the job. |
| Model fails to load | Check memory/context, compatible projector/MTP files, runtime capabilities, and `logs/backend.log`. Registration alone does not verify inference. |
| Local port already in use | Use an unused local port in the printed SSH command and open its matching URL, or choose another `connection.local_port` for a new session. Browser history at the old localhost port remains under that origin. |
| Reverse tunnel fails | Inspect `logs/reverse-ssh.log`. Check the fixed relay, known host, batch authentication, forwarding policy and port availability. Use `ood` or `local-forward` if compute-to-relay SSH is blocked. |
| Old job asks for Basic Auth | Stop it and start a new session. An already-running job does not receive updated login code. |
| Files panel has no workspace | Add an allowed shared-storage path to the selected profile, create that directory, then start a new job. Keep application state outside it. |

`./hpc-llm doctor` checks the installation, scheduler clients, runtime and model paths. Diagnostics and session access files are private; redact credentials before sharing logs.

## Verification

[Verification and cluster smoke tests](docs/verification.md) describe the deterministic backend tests, browser checks, and site checks that require a real allocation. Local fixtures exercise authentication, prefix handling, streaming, tools and actual filesystem writes; they do not establish GPU inference, site SSH policy or the external OOD route.

Build a source archive with:

```bash
.venv/bin/python scripts/package_release.py --output ../hpc-llm-0.2.0-source.zip
```
