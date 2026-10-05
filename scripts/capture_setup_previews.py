#!/usr/bin/env python3
"""Capture the real setup screens with a local profile store and HTTP fixtures."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import struct
from pathlib import Path
import tempfile
import threading
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from unittest.mock import patch

from textual.widgets import Button, Input, Select, TabbedContent

from hpc_llm.contracts import InferenceSettings, ModelSpec, ResourceRequest, SiteProfile
from hpc_llm.models import ModelLibrary
from hpc_llm.profiles import ProfileStore
from hpc_llm.ui import AdvancedDialog, LauncherApp


async def activate(pilot, screen, selector):
    button = screen.query_one(selector, Button)
    button.focus()
    button.scroll_visible(immediate=True)
    await pilot.pause(.08)
    await pilot.press("enter")
    await pilot.pause(.15)


async def capture(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hpc-llm-setup-preview-") as temporary:
        root = Path(temporary)
        bundled = root / "installation" / "profiles"
        bundled.mkdir(parents=True)
        profile = SiteProfile(
            description="Controlled setup fixture; cluster settings need site verification.",
            resource_presets={
                "Short GPU session": ResourceRequest(walltime="01:00:00"),
                "Overnight A100": ResourceRequest(partition="gpu", gpu_type="A100", walltime="08:00:00"),
            },
        )
        (bundled / "wehi.json").write_text(profile.model_dump_json())
        (bundled / "example-pbs.json").write_text(SiteProfile(name="example-pbs", scheduler="pbs", resources=ResourceRequest(partition="workq")).model_dump_json())
        store = ProfileStore(root / "state", root / "installation")
        store.initialize()
        profile = store.load()
        library = ModelLibrary(root / "registry", cache_dir=profile.model_cache)
        model = ModelSpec(name="DEMO · synthetic research assistant", path="fixture.gguf", supported_context=32768, size_bytes=4 * 1024**3, quantization="Q4_K_M")
        sessions = [
            dict(id="a" * 32, model=model.model_dump(), created_at=1000, started_at=1000, scheduler_state="RUNNING", backend_state="ready", job_id="1001"),
            dict(id="b" * 32, model=model.model_dump(), created_at=900, started_at=None, scheduler_state="PENDING", backend_state="stopped", job_id="1002"),
            dict(id="c" * 32, model=dict(model.model_dump(), name="DEMO · synthetic research assistant · saved research session"), created_at=1100, started_at=1100, scheduler_state="RUNNING", backend_state="stopped", demo=True),
        ]
        for size, filename in [((140, 45), "launcher-sessions.svg"), ((80, 24), "launcher-sessions-80.svg"), ((80, 64), "launcher-tall.svg")]:
            app = LauncherApp([model], sessions, profile, model_service=library, profile_service=store)
            async with app.run_test(size=size) as pilot:
                await pilot.pause(.2)
                for selector in ("#start", "#site-setup", "#launch-exit"):
                    button = app.query_one(selector)
                    assert button.region.bottom <= app.query_one("#launch").content_region.bottom
                app.save_screenshot(filename, str(destination))
                if size == (80, 24):
                    field = app.query_one("#launch-walltime", Input)
                    field.value = "02:00:00"
                    field.focus()
                    app.query_one("#launch-resource-labels").scroll_visible(immediate=True)
                    app.query_one("#launch-resource-row").scroll_visible(immediate=True)
                    await pilot.pause(.2)
                    app.save_screenshot("launcher-time.svg", str(destination))
                await pilot.press("ctrl+d")

        app = LauncherApp([model], sessions, profile, model_service=library, profile_service=store)
        async with app.run_test(size=(118, 48)) as pilot:
            await activate(pilot, app, "#site-setup")
            app.save_screenshot("site-setup.svg", str(destination))
            setup = app.screen
            await activate(pilot, setup, "#site-defaults")
            app.save_screenshot("site-defaults.svg", str(destination))
            await pilot.press("escape")
            await pilot.pause(.2)
            await pilot.click("#--content-tab-site-presets-tab")
            await pilot.pause(.2)
            setup.query_one("#site-preset", Select).value = "Overnight A100"
            await pilot.pause(.2)
            assert setup.query_one(TabbedContent).active == "site-presets-tab"
            app.save_screenshot("site-presets.svg", str(destination))
            await pilot.press("escape")
            await pilot.press("ctrl+d")

        # The ordinary ModelLibrary resolves metadata through controlled local HTTP.
        # No model files are downloaded for this preview.
        def fixture_gguf(architecture):
            def string(value):
                encoded = value.encode()
                return struct.pack("<Q", len(encoded)) + encoded
            return b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + string("general.architecture") + struct.pack("<I", 8) + string(architecture)
        files = {"Fixture-Q4_K_M.gguf": fixture_gguf("qwen3"), "mmproj-F16.gguf": fixture_gguf("clip"), "mtp-Fixture.gguf": fixture_gguf("qwen3")}
        registered_path = root / "Fixture-Q4_K_M.gguf"
        registered_path.write_bytes(files[registered_path.name])
        registered = library.register(registered_path, repo_id="controlled/preview-GGUF", revision="c" * 40)
        metadata = {"sha": "c" * 40, "siblings": [{"rfilename": name, "lfs": {"size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}} for name, raw in files.items()]}
        observed = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                observed.append(self.path)
                if not self.path.startswith("/api/models/"):
                    self.send_error(404)
                    return
                body = json.dumps(metadata).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def local_transport(request, timeout):
            remote = urlsplit(request if isinstance(request, str) else request.full_url)
            assert remote.hostname == "huggingface.co"
            return urlopen(Request(f"http://127.0.0.1:{server.server_port}{remote.path}?{remote.query}"), timeout=timeout)

        try:
            with patch("hpc_llm.models.urlopen", local_transport):
                app = LauncherApp([model], sessions, profile, model_service=library, profile_service=store)
                async with app.run_test(size=(118, 54)) as pilot:
                    await activate(pilot, app, "#import-model")
                    dialog = app.screen
                    dialog.query_one("#model-repo", Input).value = "https://huggingface.co/controlled/preview-GGUF"
                    dialog.query_one("#model-quant", Input).value = "Q4_K_M"
                    await activate(pilot, dialog, "#model-list")
                    assert dialog.install_plan["projector_filename"] == "mmproj-F16.gguf"
                    dialog.query_one("#model-download").scroll_visible(immediate=True)
                    await pilot.pause(.2)
                    app.save_screenshot("model-install-plan.svg", str(destination))
                    assert not any("/resolve/" in path for path in observed)
                    await pilot.press("escape")
                    await pilot.press("ctrl+d")
                app = LauncherApp([registered], [], profile, model_service=library)
                async with app.run_test(size=(100, 38)) as pilot:
                    await activate(pilot, app, "#import-model")
                    dialog = app.screen
                    dialog.query_one("#model-target", Select).value = registered.id
                    await pilot.pause(.2)
                    dialog.query_one("#model-remote-projector", Select).value = "auto"
                    dialog.query_one("#model-remote-mtp", Select).value = "auto"
                    await activate(pilot, dialog, "#model-list")
                    assert {item["filename"] for item in dialog.install_plan["files"]} == {"mmproj-F16.gguf", "mtp-Fixture.gguf"}
                    dialog.query_one("#model-download").scroll_visible(immediate=True)
                    await pilot.pause(.2)
                    app.save_screenshot("model-companions.svg", str(destination))
                    dialog.query_one("#model-verify-full").focus()
                    dialog.query_one("#model-verify-full").scroll_visible(immediate=True, top=True)
                    await pilot.pause(.2)
                    app.save_screenshot("model-verification.svg", str(destination))
                    assert not any("/resolve/" in path for path in observed)
                    await pilot.press("escape")
                    app.push_screen(AdvancedDialog(InferenceSettings(acceleration="auto").model_dump(), {
                        "acceleration_status": "Unavailable",
                        "acceleration_reason": "No matching MTP head is registered in this allocation",
                        "runtime_controls": ["acceleration", "spec_draft_n_max", "spec_draft_n_min", "spec_draft_p_min"],
                    }, ResourceRequest().model_dump()))
                    await pilot.pause(.2)
                    app.screen.query_one(TabbedContent).active = "runtime-tab"
                    await pilot.pause(.2)
                    app.save_screenshot("advanced-mtp.svg", str(destination))
                    await pilot.press("escape")
                    await pilot.press("ctrl+d")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    asyncio.run(capture(args.output.resolve()))
