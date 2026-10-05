#!/usr/bin/env python3
"""Capture actual Textual screens through the real supervisor and demo backend."""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import tempfile

from textual.widgets import Collapsible, Input, TabbedContent, TextArea

from hpc_llm.client import SessionClient
from hpc_llm.contracts import InferenceSettings, ModelSpec, ResourceRequest, SiteProfile
from hpc_llm.lifecycle import SessionManager
from hpc_llm.supervisor import Supervisor
from hpc_llm.ui import AdvancedDialog, ChatApp, SourceSelectionDialog


async def capture(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hpc-llm-previews-") as temp:
        workspace = Path(temp)
        (workspace / "research-notes.txt").write_text("Synthetic pilot study\nTreatment A: 12 units, 3 replicates.\nTreatment B: 18 units, 3 replicates.\nNo experimental claims: this is a controlled fixture.\n")
        (workspace / "measurements.csv").write_text("treatment,value\nA,11\nA,12\nA,13\nB,17\nB,18\nB,19\n")
        (workspace / "documents").mkdir()
        previous = Path.cwd()
        os.chdir(workspace)
        service = None
        try:
            manager = SessionManager(workspace / "private-state", workspace / "hpc-llm")
            manifest = manager.create(ModelSpec(name="Research assistant · synthetic demo", path="demo.gguf", supported_context=32768), ResourceRequest(), SiteProfile(modules=[]), InferenceSettings(web_enabled=True, web_provider="fixture", web_approval="per_request"), demo=True)
            service = Supervisor(manifest)
            await service.start()
            app = ChatApp(SessionClient(service.manifest))
            async with app.run_test(size=(118, 36)) as pilot:
                await pilot.pause(.2)
                await pilot.press("ctrl+o")
                await pilot.pause(.1)
                app.screen.query_one("#attach-path", Input).value = str(workspace / "research-notes.txt")
                await pilot.click("#attach-apply")
                await pilot.pause(.3)
                await pilot.press("escape")
                app.query_one("#composer", TextArea).load_text("Compare Treatment A and Treatment B and cite the local notes.")
                await pilot.press("ctrl+s")
                await pilot.pause(.8)
                app.save_screenshot("chat.svg", str(destination))
                # File panel is populated by actual server-side directory listing.
                await pilot.click("#attachments")
                await pilot.pause(.2)
                for disclosure in app.screen.query(Collapsible):
                    disclosure.collapsed = False
                await pilot.pause(.2)
                app.save_screenshot("files.svg", str(destination))
                await pilot.press("escape")
                app.push_screen(AdvancedDialog(app.manifest["settings"], app.manifest["capabilities"], app.manifest["resources"]))
                await pilot.pause(.2)
                app.save_screenshot("advanced.svg", str(destination))
                await pilot.press("escape")
                await pilot.click("#thinking")
                await pilot.pause(.2)
                await pilot.click("#thinking-value")
                await pilot.pause(.2)
                app.save_screenshot("thinking.svg", str(destination))
                await pilot.press("escape")
                await pilot.press("escape")
                approval = await app.api("POST", "/tools/request", json={"conversation_id": app.conversation_id, "name": "web_search", "arguments": {"query": "synthetic protein research"}})
                # Capture exact-action review from the real persisted approval.
                await app.refresh_state()
                await pilot.pause(.2)
                app.save_screenshot("web-approval.svg", str(destination))
                await pilot.click("#approval-yes")
                await pilot.pause(.8)
                if isinstance(app.screen, SourceSelectionDialog):
                    disclosures = list(app.screen.query(Collapsible))
                    if disclosures:
                        disclosures[0].collapsed = False
                    await pilot.pause(.2)
                    app.save_screenshot("web-sources.svg", str(destination))
                    await pilot.press("escape")
                await app.action_new_chat()
                app.query_one("#composer", TextArea).load_text("Markdown preview\n\n## Research plan\n\n**Synthetic results**\n\n| Treatment | Units |\n| --- | --- |\n| A | 12 |\n| B | 18 |\n\n```python\nprint('fixture')\n```")
                await pilot.press("ctrl+s")
                await pilot.pause(.9)
                await app.refresh_state()
                turn_id = app.conversation["turns"][-1]["id"]
                app.query_one(f"#open-reply-{turn_id}").scroll_visible(animate=False)
                await pilot.pause(.2)
                await pilot.click(f"#open-reply-{turn_id}")
                await pilot.pause(.2)
                app.save_screenshot("reply-reader.svg", str(destination))
                app.screen.query_one("#reply-views", TabbedContent).active = "reply-raw-tab"
                await pilot.pause(.2)
                app.save_screenshot("reply-raw.svg", str(destination))
                await pilot.click("#reply-save")
                app.screen.query_one("#reply-destination", Input).value = "research-brief.md"
                await pilot.pause(.2)
                app.save_screenshot("save-reply.svg", str(destination))
                await pilot.click("#save-reply-confirm")
                await pilot.pause(.4)
                await app.refresh_state()
                app.query_one("#transcript").scroll_end(animate=False)
                await pilot.pause(.2)
                app.save_screenshot("saved-reply.svg", str(destination))
                await app.action_new_chat()
                app.query_one("#composer", TextArea).load_text("[demo-write-file] Write a Markdown file in the workspace; report only the saved path.")
                await pilot.press("ctrl+s")
                await pilot.pause(.9)
                await app.refresh_state()
                app.save_screenshot("generated-file.svg", str(destination))
                await pilot.resize_terminal(80, 24)
                await pilot.pause(.2)
                app.save_screenshot("chat-80x24.svg", str(destination))
                await pilot.press("ctrl+d")
        finally:
            if service:
                await service.close()
            os.chdir(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("assets"))
    args = parser.parse_args()
    asyncio.run(capture(args.output.resolve()))
