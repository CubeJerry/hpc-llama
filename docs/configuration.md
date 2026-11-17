# Configuration

`config.yaml` is the saved source of truth for profiles. Version 1 supports Slurm and PBS Pro / OpenPBS. Unknown fields, unsupported schema versions, duplicate YAML keys, aliases, custom tags, control characters and files above 1 MiB are rejected. Parsing does not execute YAML objects or shell expressions.

## Select the configuration and paths

The configuration path is selected in this order:

1. Global `--config PATH`, before the subcommand.
2. `HPC_LLM_CONFIG`.
3. `config.yaml` under the initial state directory, normally `state/config.yaml` inside the installation.

The initial state directory comes from `--state-root PATH`, then `HPC_LLM_HOME`, then the installation's `state` directory. If no `--state-root` is given, a nonempty `state_directory` in the selected YAML chooses the state location. Selecting a state directory does not move existing models, registrations or sessions: copy them yourself if relocating, between jobs. Keep the selected configuration path available when reconnecting. To manage existing sessions despite a broken live configuration, pass `--state-root` explicitly to `sessions`, `attach` or `stop`; these commands use the saved session snapshot.

Profile paths expand `~` and environment variables such as `$USER`. Relative model-cache, runtime, workspace and `state_directory` paths start at the installation directory. Unset variables are errors. Keep state, model cache and workspaces on storage shared with compute nodes; local `/tmp` usually does not qualify.

| Setting | Precedence, highest first |
|---|---|
| Model cache | `HPC_LLM_MODEL_CACHE` → top-level `model_cache` → selected profile's `model_cache` → `cache/models` |
| Runtime executable | `HPC_LLM_RUNTIME` → selected profile's `runtime` → `runtime/bin/llama-server` |
| Resource request | Explicit launch CLI overrides → one-session launcher edits → selected preset or `default_preset` → profile's `resources` |

CLI launch overrides stay visible in the launcher and take priority over its fields. Session changes do not rewrite presets. **Site setup… → Save & use**, `profiles use` and `profiles preset` deliberately change saved configuration. A global cache override remains active when selecting another profile.

```bash
./hpc-llm --config /absolute/path/config.yaml profiles show
./hpc-llm profiles list
./hpc-llm profiles use slurm
./hpc-llm profiles cache /absolute/shared/model-cache
./hpc-llm launch --preset short --partition gpu-general --walltime 00:45:00
```

## Schema version 1

Start from [the Slurm example](../examples/config.slurm.yaml). New installations select `slurm`, which uses local SSH forwarding until you choose another route. Set the partition, GPU label and SSH hosts for your cluster. The optional [WEHI example](../examples/config.wehi.yaml) supplies Milton settings and uses OOD; `wehi` runs through the same Slurm scheduler as `slurm`.

Running `bash install.sh` on an existing installation keeps its selected profile and custom settings. Use `bash install.sh --profile slurm` or `./hpc-llm profiles use slurm` to switch. Bundled profiles remain selectable even if an older YAML file does not contain them yet; selecting one saves it to YAML. Existing YAML entries take precedence over bundled templates. The previous `generic` profile remains supported.

| Top-level field | Meaning |
|---|---|
| `schema_version` | Required; `1`. |
| `selected_profile` | The key of an entry in `profiles`. |
| `profiles` | Mapping of site names to profiles; names use 1–64 letters, digits, `_` or `-`, starting with a letter or digit. |
| `state_directory` | Optional application-state location; empty preserves the initial state directory. |
| `model_cache` | Optional cache override across profiles. |

| Profile field | Meaning |
|---|---|
| `name` | Optional in hand-written YAML; when present, must match its mapping key. |
| `description` | A label for the site. |
| `scheduler` | `slurm` or `pbs`. |
| `resources` | Default request, described below. |
| `resource_presets` | Named complete resource requests. |
| `default_preset` | Optional name from `resource_presets`. |
| `model_cache` | Site-specific weight/companion cache, unless globally overridden. |
| `runtime` | One executable path; empty selects the managed runtime. |
| `modules` | List of literal module names, such as `cuda/12.4`. |
| `runtime_environment` | Mapping of environment names to printable literal strings. Values are quoted when exported, never evaluated as shell code. |
| `workspaces` | Allowed shared-storage directories for browser file access. Select the active workspace in the launcher. |
| `connection` | Browser route configuration, described below. |
| `tools` | Browser file and tool settings, described below. |
| `pbs_gpu_resource` | PBS count resource, normally `ngpus`; set the site's name. |
| `pbs_gpu_type_resource` | PBS GPU-type resource; required if requesting a nonempty `gpu_type`. |
| `pbs_attach_command` | One executable name/path, normally `pbs_attach`, for retained legacy terminal sessions. |

`modules` requires an already-initialised `module` command in the batch shell. Arbitrary `module_init` shell setup is rejected. Use an explicit executable wrapper if a runtime requires more setup. The wrapper is code you install and review, rather than code evaluated from YAML. Scheduler environment overrides and shell startup hooks are rejected in `runtime_environment`. Do not store credentials there.

### Resource requests

| Field | Meaning and limits |
|---|---|
| `partition` | Nonempty Slurm partition or PBS queue, entered manually. |
| `gpu_type` | Site GPU/GRES label; empty requests a generic GPU. |
| `gpu_count` | Exactly `1`; multi-GPU requests are rejected. |
| `cpus` | 1–128 CPU cores. |
| `memory_gb` | 1–2048 GB of host RAM; not GPU VRAM. |
| `walltime` | Positive `HH:MM:SS` or `D-HH:MM:SS`; quote it in YAML. |
| `account` | Optional allocation account/project. |
| `qos`, `constraint` | Optional Slurm fields; PBS rejects them. |

Partition, GPU, account and expression fields accept scheduler-safe literal characters, not spaces or shell commands. They are not site-specific enums. Inventory discovery helps when `sinfo` or PBS queue information is available; restricted discovery does not prevent manual configuration.

Presets are **complete requests**. Omitted preset fields use schema defaults rather than inheriting individual fields from `resources`. Include all values that matter, especially account, memory, CPU count and GPU label. Selecting another preset changes the request for that launch; editing the request does not save it back to the preset.

Slurm needs `sbatch`, `squeue`, `sacct` and `scancel`. PBS needs `qsub`, `qdel`, `qstat -fx -F json` and retained job history. GPU resource naming and accounting policies need site validation. Other workload managers are unsupported.

## Connection settings

The GPU job owns the authenticated entry point in all three modes. The launcher only submits the job and shows its connection details. Readiness checks the backend, cookie login and model API. It cannot check your computer's SSH reachability or the site's external OOD route.

### Open OnDemand

```yaml
connection:
  mode: ood
  ood_origin: https://ondemand.example.edu
  ood_path: /rnode/{node}/{port}
```

`ood_origin` is an HTTPS origin without a path, credentials, query or fragment. `ood_path` is the site's approved route with exactly one `{node}` and `{port}` placeholder. The job chooses a free port and binds its authenticated entry point for the compute proxy. Inference stays on loopback. No SSH tunnel is involved.

### Local SSH forwarding

```yaml
connection:
  mode: local-forward
  ssh_host: "user@{node}"
  jump_host: cluster-login
  local_port: 8765
```

Run the printed SSH command on your computer. The SSH destination must reach the compute node itself; `jump_host` is the optional approved jump host. `{node}` expands to the allocated node. The service and local forward bind loopback. Keep the workstation SSH command running while using chat. The GPU job survives that command closing.

### Reverse SSH forwarding

```yaml
connection:
  mode: reverse-ssh
  relay_host: fixed-login-01
  relay_jump_host: ""
  jump_host: workstation-jump
  relay_port: 0
  local_port: 8765
```

The compute job opens the reverse forward to `relay_host`. Its `relay_jump_host` is optional and applies only to that compute-to-relay connection. Separately, your computer runs the printed local forward to the same relay; `jump_host` applies to this workstation connection. These aliases must identify the same concrete host from both sides. A load-balanced login alias is unsuitable.

`relay_port: 0` asks OpenSSH to allocate a free relay port; an explicit occupied port fails rather than falling back silently. Relay and workstation forward endpoints remain loopback. The job verifies the reverse path with an authenticated round trip before publishing it. An unexpected compute-to-relay tunnel disconnect fails the service and releases the allocation. Local browser disconnection does not. Inspect `logs/reverse-ssh.log` for startup failures.

All ports are 1024–65535, except automatic `relay_port: 0`. SSH destinations are one hostname/alias, optionally `user@host`. Configure SSH ports in your own SSH configuration. Forwarding uses batch authentication, strict host-key checking, startup failure detection and keepalives. Provision credentials and known hosts yourself; the app does not change them. Choose another mode when the site's forwarding policy or compute-to-relay connectivity prohibits this route.

Launch flags override connection settings for one session:

```bash
./hpc-llm launch --connection ood --ood-url https://ondemand.example.edu
./hpc-llm launch --connection local-forward --ssh-host '{node}' --jump-host cluster-login
./hpc-llm launch --connection reverse-ssh --relay-host fixed-login-01 --local-port 8766
```

`attach SESSION_ID` redisplays the saved route without resubmitting or rewriting it. To change startup connection settings or a saved local port, start a new session; you can also choose an unused local port manually in the printed workstation command and open the matching URL.

## Workspaces and tool limits

The launcher lets you select one workspace for this session. When profile `workspaces` are configured, the selected path must be inside one of them. With no configured roots, an explicit launcher path becomes this session's workspace. The launcher prefills the installation's `workspace` directory as the default and creates that default automatically. A custom directory must already exist; an empty path is rejected. Each job saves the allowed scope with its session configuration.

Use a dedicated directory for inputs and outputs. Keep application state, private SSH files and secrets outside it. File access rejects traversal and symlink escapes; it does not run shell commands. The panel reads UTF-8 text, code, CSV and JSON. Binary documents and media belong in native attachments when the model/UI supports them.

| `tools` field | Default | Allowed range or effect |
|---|---|---|
| `enabled` | `true` | Enables the integrated browser files/tools panel. |
| `allow_writes` | `false` | `false` previews and approves each model write/overwrite; `true` deliberately permits model writes inside the selected workspace without per-write approval. Manual editor saves still use a preview. |
| `max_file_bytes` | `262144` | 1024–2097152 bytes per file. |
| `max_entries` | `500` | 1–5000 directory entries. |
| `max_tool_calls` | `8` | 1–16 calls per model-driven file task. |
| `timeout_seconds` | `120` | 5–300 seconds per task, including time awaiting approval. |

The panel runs at most two model file tasks concurrently. Its preview tokens expire after 15 minutes and can be used once. A write is refused if the destination changed since its preview. Cancel a task in the panel to end its loop. Model requests, network waits and approval waits are cancellable; a timeout cannot forcibly interrupt a stalled operating-system/filesystem call. File-size and directory-entry limits bound normal filesystem work, and cancellation does not leave background writes running. Task activity is transient GPU-process state; download its JSON transcript before the job ends.

## JSON migration and rollback

If no canonical YAML exists, the first profile access imports the bundled profiles, managed `state/profiles/*.json` overlays and `state/config.json` profile/cache selection. Existing JSON originals stay intact. Copies go into `state/migration-backup-v1/`, or a new suffixed directory if that backup already exists. From then on, the YAML is authoritative: editing old JSON does not update it.

Model registrations, exact companion paths, per-model defaults, cached weights and saved session snapshots live in other files and are not rewritten during this migration. Unknown legacy settings or unsupported shell setup produce an error instead of being silently dropped. Fix the reported field or migrate it manually while retaining the originals.

To return to the preserved `tui` version, stop modern jobs, keep a copy of modern YAML and state, then use the original retained JSON or the migration backup with that branch's installer. Do not replace the whole state directory with the configuration backup: it contains configuration files, not models or conversation snapshots. Moving YAML aside and starting the modern version again reimports the retained JSON; it does not roll back runtime code.

## Design references

The transports use standard site-approved forwarding and proxy routes. These references describe the mechanisms and scheduler interfaces:

- [OpenSSH `ssh`](https://man.openbsd.org/ssh.1): `-L`, `-R`, jump hosts and forwarding controls.
- [OpenSSH `ssh_config`](https://man.openbsd.org/ssh_config.5): batch mode, host-key verification, forwarding failure and keepalive options.
- [Open OnDemand interactive app routes](https://osc.github.io/ood-documentation/release-1.8/app-development/interactive/view.html) and [portal proxy template](https://github.com/OSC/ondemand/blob/master/ood-portal-generator/templates/ood-portal.conf.erb): compute-node web access through the site's web portal, including `/rnode` and `/node` path handling.
- [Slurm `sbatch`](https://slurm.schedmd.com/sbatch.html): resource and accounting directives.
- [CryoSPARC access guide](https://guide.cryosparc.com/setup-configuration-and-management/how-to-download-install-and-configure/accessing-cryosparc): local SSH forwarding and reverse-proxy access as separate choices. HPC Llama does not claim CryoSPARC uses compute-to-relay reverse SSH.
