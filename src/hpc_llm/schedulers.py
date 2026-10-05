"""Scheduler adapters for one persistent Slurm or PBS Pro/OpenPBS allocation.

Only scheduler commands and attach transport differ. Requests remain authenticated
compute-local HTTP; no files are used as a request queue.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import socket

from .contracts import AppError, ENDED_SCHEDULER_STATES, ResourceRequest, SessionManifest, SiteProfile


def current_job_id(profile: SiteProfile) -> str | None:
    return os.environ.get("PBS_JOBID" if profile.scheduler == "pbs" else "SLURM_JOB_ID") or None


def inside_allocation(manifest: SessionManifest) -> bool:
    if not manifest.job_id or current_job_id(manifest.profile) != manifest.job_id:
        return False
    if manifest.profile.scheduler == "slurm":
        return True
    if not manifest.node:
        return False
    local = {socket.gethostname().lower().rstrip("."), socket.getfqdn().lower().rstrip(".")}
    return manifest.node.lower().rstrip(".") in local


def _job_name(manifest: SessionManifest) -> str:
    return f"hpc-llm-{manifest.id[:12]}"


def _terminal(state: str) -> bool:
    return (state.split() or [""])[0].rstrip("+") in ENDED_SCHEDULER_STATES


def _unknown(manifest: SessionManifest, reason: str) -> SessionManifest:
    manifest.scheduler_state = "UNKNOWN"
    manifest.scheduler_reason = reason
    return manifest


def _paths(manifest, launcher, root):
    r = ResourceRequest.model_validate(manifest.resources.model_dump())
    if r.gpu_count != 1:
        raise AppError("unsupported", "Only a single GPU and single compute node are supported")
    for path in (Path(manifest.directory), launcher, root):
        if not Path(path).is_absolute() or any(ord(c) < 32 or ord(c) == 127 for c in str(path)):
            raise AppError("validation", "Scheduler paths must be absolute and contain no control characters")
    return r


def _command(manifest, launcher, root, mode):
    return [str(launcher), "--state-root", str(root), mode, manifest.id]


class SlurmScheduler:
    name = "slurm"

    def render_batch(self, manifest, launcher, root):
        r = _paths(manifest, launcher, root)
        folder = Path(manifest.directory)
        gpu_resource = f"gpu:{r.gpu_type}:{r.gpu_count}" if r.gpu_type else f"gpu:{r.gpu_count}"
        directives = [f"#SBATCH --job-name={_job_name(manifest)}", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
                      f"#SBATCH --partition={r.partition}", f"#SBATCH --gres={gpu_resource}",
                      f"#SBATCH --cpus-per-task={r.cpus}", f"#SBATCH --mem={r.memory_gb}G", f"#SBATCH --time={r.walltime}",
                      f"#SBATCH --output={shlex.quote(str(folder / 'logs' / 'allocation-%j.log'))}",
                      f"#SBATCH --chdir={shlex.quote(str(launcher.parent))}", "#SBATCH --signal=B:TERM@30"]
        for field in ("account", "qos", "constraint"):
            if getattr(r, field):
                directives.append(f"#SBATCH --{field}={getattr(r, field)}")
        return "\n".join(["#!/bin/bash", *directives, "set -euo pipefail", "umask 077",
                           "exec " + shlex.join(_command(manifest, launcher, root, "_supervisor"))]) + "\n"

    def submit(self, script, manifest, run):
        result = run(["sbatch", "--parsable", str(script)])
        if result.returncode:
            raise AppError("scheduler", "Slurm rejected the allocation; check partition, resources and account settings")
        match = re.fullmatch(r"\s*(\d+)(?:;([A-Za-z0-9_.-]+))?\s*", result.stdout)
        if not match:
            raise AppError("scheduler", "Slurm returned an unexpected job identifier; inspect submitted jobs before retrying")
        return match[1], match[2]

    def status(self, manifest, run):
        try:
            result = run(["squeue", "-h", "-j", manifest.job_id, "-o", "%i|%T|%R|%N|%L|%u|%j"])
            if result.returncode:
                return _unknown(manifest, "Scheduler unavailable; allocation state is unknown")
            lines = result.stdout.strip().splitlines()
            if lines:
                parts = lines[0].split("|")
                if len(parts) >= 7 and parts[0] == manifest.job_id:
                    if parts[5] != manifest.owner or parts[6] != _job_name(manifest):
                        raise AppError("permission", "Job identity no longer matches this session")
                    manifest.scheduler_state, manifest.scheduler_reason = parts[1], parts[2]
                    if parts[3] not in {"", "(null)"}:
                        manifest.node = parts[3]
                    return manifest
            accounting = run(["sacct", "-n", "-X", "-j", manifest.job_id, "--parsable2", "--format=JobID,State,User,JobName"])
            for line in accounting.stdout.splitlines():
                parts = line.split("|")
                if len(parts) >= 4 and parts[0] == manifest.job_id:
                    if parts[2] != manifest.owner or parts[3] != _job_name(manifest):
                        raise AppError("permission", "Accounting job identity does not match")
                    manifest.scheduler_state = parts[1].split()[0]
                    manifest.scheduler_reason = ""
                    return manifest
            return _unknown(manifest, "Waiting for scheduler accounting; state is unknown")
        except AppError as exc:
            if exc.code == "permission":
                raise
            return _unknown(manifest, "Scheduler query failed; allocation state is unknown")

    def attach_argv(self, manifest, launcher, root, run):
        if not manifest.job_id or not manifest.job_id.isdigit():
            raise AppError("scheduler", "Session has no allocation to attach to")
        return ["srun", f"--jobid={manifest.job_id}", "--overlap", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--pty",
                *_command(manifest, launcher, root, "_tui")]

    def stop(self, manifest, run):
        if _terminal(manifest.scheduler_state):
            return
        if manifest.scheduler_state == "UNKNOWN":
            raise AppError("scheduler", "Cannot verify allocation ownership; retry after scheduler access returns")
        result = run(["scancel", manifest.job_id])
        if result.returncode:
            raise AppError("scheduler", "Slurm could not stop this allocation; inspect sessions and retry")

    def inventory(self, profile, run):
        result = run(["sinfo", "-h", "-o", "%P|%G|%c|%m"], timeout=10)
        if result.returncode:
            raise AppError("scheduler", "Read-only Slurm inventory unavailable")
        partitions, seen = [], set()
        for line in result.stdout.splitlines()[:1000]:
            fields = line.split("|")
            if len(fields) != 4 or not fields[2].isdigit() or not fields[3].isdigit():
                continue
            partition = fields[0].rstrip("*")
            if not re.fullmatch(r"[A-Za-z0-9_.+-]+", partition) or tuple(fields) in seen:
                continue
            seen.add(tuple(fields))
            partitions.append({"name": partition, "default": fields[0].endswith("*"), "gres": fields[1][:512],
                               "cpus_per_node": int(fields[2]), "memory_mb_per_node": int(fields[3])})
        return {"available": bool(partitions), "partitions": partitions, "scheduler": "slurm"}


def _pbs_id(job_id: str | None) -> str:
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9]+(?:\.[A-Za-z0-9][A-Za-z0-9_.-]*)?", job_id):
        raise AppError("scheduler", "Invalid PBS allocation identifier; job arrays are not supported")
    return job_id


def _pbs_hostname(job: dict) -> str | None:
    # exec_host/exec_host2 are server-reported execution hosts, unlike exec_vnode,
    # which may contain abstract vnode names unsuitable as SSH destinations.
    specification = job.get("exec_host2") or job.get("exec_host")
    if not specification:
        return None
    if not isinstance(specification, str) or len(specification) > 10000:
        raise AppError("permission", "PBS returned an invalid execution host")
    hosts = set()
    for chunk in specification.split("+"):
        host = chunk.split("/", 1)[0].split(":", 1)[0]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,252}", host):
            raise AppError("permission", "PBS returned an unsafe execution host")
        hosts.add(host)
    if len(hosts) != 1:
        raise AppError("unsupported", "This PBS session requires exactly one execution host")
    return hosts.pop()


class PBSScheduler:
    name = "pbs"

    def render_batch(self, manifest, launcher, root):
        r = _paths(manifest, launcher, root)
        p = SiteProfile.model_validate(manifest.profile.model_dump())
        if r.qos or r.constraint:
            raise AppError("unsupported", "PBS does not use Slurm QOS/constraint fields; clear them and configure the PBS queue/resources")
        if not p.pbs_gpu_resource:
            raise AppError("validation", "Configure the site's PBS GPU-count resource, normally ngpus")
        if r.gpu_type and not p.pbs_gpu_type_resource:
            raise AppError("validation", "A PBS GPU type requires the site's configured GPU-type resource name; configure it or clear GPU type")
        select = f"select=1:ncpus={r.cpus}:mem={r.memory_gb}gb:{p.pbs_gpu_resource}={r.gpu_count}"
        if r.gpu_type:
            select += f":{p.pbs_gpu_type_resource}={r.gpu_type}"
        day, clock = r.walltime.split("-", 1) if "-" in r.walltime else ("0", r.walltime)
        hours, minutes, seconds = clock.split(":")
        walltime = f"{int(day)*24+int(hours)}:{minutes}:{seconds}"
        folder = Path(manifest.directory)
        directives = [f"#PBS -N {_job_name(manifest)}", f"#PBS -q {r.partition}", f"#PBS -l {select}", f"#PBS -l walltime={walltime}",
                      "#PBS -j oe", f"#PBS -o {shlex.quote(str(folder / 'logs' / 'allocation.log'))}"]
        if r.account:
            directives.append(f"#PBS -A {r.account}")
        return "\n".join(["#!/bin/bash", *directives, "set -euo pipefail", "umask 077", "cd -- " + shlex.quote(str(launcher.parent)),
                           "exec " + shlex.join(_command(manifest, launcher, root, "_supervisor"))]) + "\n"

    def submit(self, script, manifest, run):
        result = run(["qsub", "-C", "#PBS", str(script)])
        if result.returncode:
            raise AppError("scheduler", "PBS rejected the allocation; check queue, account and the site's GPU resource names")
        job_id = _pbs_id(result.stdout.strip())
        return job_id, job_id.partition(".")[2] or None

    def status(self, manifest, run):
        job_id = _pbs_id(manifest.job_id)
        try:
            result = run(["qstat", "-fx", "-F", "json", job_id])
            if result.returncode:
                return _unknown(manifest, "PBS query unavailable or history delayed; allocation state is unknown")
            if len(result.stdout) > 4 * 1024 * 1024:
                raise AppError("scheduler", "PBS status response exceeded its safety limit")
            listing = json.loads(result.stdout)
            jobs = listing.get("Jobs", {})
            job = jobs.get(job_id)
            if not isinstance(job, dict):
                return _unknown(manifest, "PBS accounting has no matching allocation; state is unknown")
            owner = str(job.get("Job_Owner", "")).split("@", 1)[0]
            if owner != manifest.owner or job.get("Job_Name") != _job_name(manifest):
                raise AppError("permission", "PBS allocation owner or session job name no longer matches")
            state = job.get("job_state")
            if state == "F":
                exit_status = job.get("Exit_status")
                if exit_status is None:
                    return _unknown(manifest, "PBS finished allocation is waiting for exit accounting; state is unknown")
                manifest.scheduler_state = "COMPLETED" if int(exit_status) == 0 else "FAILED"
            else:
                manifest.scheduler_state = {"Q": "PENDING", "H": "HELD", "R": "RUNNING", "E": "COMPLETING",
                                            "S": "SUSPENDED", "U": "SUSPENDED", "W": "PENDING", "T": "PENDING",
                                            "B": "RUNNING", "X": "COMPLETED"}.get(state, "UNKNOWN")
            manifest.scheduler_reason = str(job.get("comment", ""))[:500]
            if manifest.scheduler_state == "UNKNOWN":
                manifest.scheduler_reason = "PBS returned an unsupported job state; state is unknown"
            node = _pbs_hostname(job)
            if node:
                manifest.node = node
            else:
                manifest.node = None
            server = listing.get("pbs_server") or job.get("server")
            if isinstance(server, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", server):
                manifest.cluster = server
            elif manifest.scheduler_state == "RUNNING":
                return _unknown(manifest, "PBS server identity is missing; attachment is unavailable")
            return manifest
        except AppError as exc:
            if exc.code in {"permission", "unsupported"}:
                raise
            return _unknown(manifest, "PBS query failed; allocation state is unknown")
        except (ValueError, TypeError, AttributeError):
            return _unknown(manifest, "PBS returned malformed status; allocation state is unknown")

    def attach_argv(self, manifest, launcher, root, run):
        checked = self.status(manifest.model_copy(deep=True), run)
        if checked.scheduler_state != "RUNNING" or not checked.node:
            raise AppError("scheduler", "PBS allocation is not verified running on one execution host. Check queue/accounting status before attaching")
        command = _command(manifest, launcher, root, "_tui")
        if inside_allocation(checked):
            return command
        # pbs_attach's documented cmd form runs its child only after attaching
        # its session to the job. PBS_JOBID is set in that child, never before it.
        remote = [manifest.profile.pbs_attach_command, "-j", _pbs_id(checked.job_id), "/usr/bin/env", f"PBS_JOBID={checked.job_id}", *command]
        return ["ssh", "-tt", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ClearAllForwardings=yes", "--", checked.node,
                "exec " + shlex.join(remote)]

    def stop(self, manifest, run):
        if _terminal(manifest.scheduler_state):
            return
        if manifest.scheduler_state == "UNKNOWN":
            raise AppError("scheduler", "Cannot verify PBS allocation ownership; retry after qstat/accounting access returns")
        result = run(["qdel", _pbs_id(manifest.job_id)])
        if result.returncode:
            raise AppError("scheduler", "PBS could not stop this allocation; inspect sessions and retry")

    def inventory(self, profile, run):
        result = run(["qstat", "-Qf", "-F", "json"], timeout=10)
        if result.returncode or len(result.stdout) > 4 * 1024 * 1024:
            raise AppError("scheduler", "Read-only PBS queue inventory unavailable")
        try:
            data = json.loads(result.stdout)
            queues = data.get("Queue", data.get("Queues", {}))
            partitions = []
            for name, values in list(queues.items())[:1000]:
                if not re.fullmatch(r"[A-Za-z0-9_.+-]+", name) or not isinstance(values, dict):
                    continue
                if values.get("queue_type", "Execution") != "Execution":
                    continue
                available = values.get("resources_available", {})
                if not isinstance(available, dict):
                    available = {}
                fields = {key: available[key] for key in ("ncpus", "mem", profile.pbs_gpu_resource, profile.pbs_gpu_type_resource) if key and key in available}
                partitions.append({"name": name, "enabled": values.get("enabled"), "started": values.get("started"), "resources_available": fields})
            return {"available": bool(partitions), "partitions": partitions, "scheduler": "pbs"}
        except (ValueError, TypeError, AttributeError):
            raise AppError("scheduler", "PBS queue inventory was not valid JSON") from None


def scheduler_for(profile: SiteProfile) -> SlurmScheduler | PBSScheduler:
    return PBSScheduler() if profile.scheduler == "pbs" else SlurmScheduler()
