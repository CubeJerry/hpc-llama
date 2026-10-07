"""Terminal UI. Every chat operation uses the authenticated supervisor client.

No scheduler command or inference stream is owned by a screen. Closing this app
only detaches; the batch-owned supervisor retains the durable conversation.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.markdown import Markdown
from textual import on
from textual.events import Resize
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.scrollbar import ScrollBar
from textual.widgets import (
    Button, Checkbox as TextualCheckbox, Collapsible, Footer, Input, Label, Select, Static,
    TabbedContent, TabPane, TextArea, Tree,
)

from .contracts import InferenceSettings, ResourceRequest, SiteProfile, session_can_resume, session_is_ended
from .portable_scrollbar import PortableScrollBarRender
from .vram import estimate_vram


# Documented renderer hook covers chat, file lists, dialogs and launcher alike.
ScrollBar.renderer = PortableScrollBarRender


# Fractional block borders render inconsistently in the observed OOD terminal.
# Use ordinary ASCII for structural borders rather than relying on those glyphs;
# source/user text keeps its Unicode characters and the colour theme is unchanged.
# !important also covers built-in hover, pressed, disabled and focus variants.
PORTABLE_BORDERS_CSS = """
Button, Input, SelectCurrent, SelectOverlay, TextArea, Checkbox {
    border: ascii $border-blurred !important;
}
Button.-primary { border: ascii $primary !important; }
Button.-error { border: ascii $error !important; }
Button:focus, Input:focus, Select:focus > SelectCurrent,
TextArea:focus, Checkbox:focus {
    border: ascii $accent !important;
}
Dialog > Vertical, #launch { border: ascii $accent !important; }
#drawer { border-right: ascii $primary-background !important; }
Collapsible { border-top: ascii $background !important; }
#tray Button { border: none !important; }
"""


class Checkbox(TextualCheckbox):
    """Keep the checkbox indicator on the same ASCII cell grid as its border."""
    BUTTON_LEFT = "["
    BUTTON_RIGHT = "]"


class ChatComposer(TextArea):
    """Chat-only keys; terminal Paste events retain TextArea's literal insert."""
    BINDINGS = [
        Binding("enter", "submit", "Send", show=False, priority=True),
        Binding("shift+enter,ctrl+j", "newline", "New line", show=False, priority=True),
    ]

    class Submitted(Message):
        pass

    def action_submit(self) -> None:
        self.post_message(self.Submitted())

    def action_newline(self) -> None:
        self.replace("\n", *self.selection, maintain_selection_offset=False)


def data(value: Any) -> Any:
    return value.model_dump() if hasattr(value, "model_dump") else value


def records(value: Any) -> list[dict]:
    if isinstance(value, dict):
        return [data(v) for v in value.values()]
    return [data(v) for v in (value or [])]


def safe(value: Any) -> str:
    """Strip ANSI/OSC and invisible controls before rendering untrusted text."""
    text = str(value or "")
    text = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", text)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return "".join(c for c in text if c in "\n\t" or (ord(c) >= 32 and ord(c) != 127 and not 0x80 <= ord(c) <= 0x9f and not 0x202a <= ord(c) <= 0x202e and not 0x2066 <= ord(c) <= 0x2069))


def plain(text: Any, **kwargs: Any) -> Static:
    return Static(safe(text), markup=False, **kwargs)


def source_text(source: dict) -> str:
    stamp = source.get("retrieved_at")
    retrieved = datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if isinstance(stamp, (int, float)) else "unknown"
    kind = {"web_snippet": "Search snippet (page not read)", "web_page": "Fetched page", "file": "Local file excerpt"}.get(source.get("kind"), "Source")
    return safe("\n".join(filter(None, [
        f"[{source.get('id', '?')}] {source.get('title', 'Untitled')}",
        f"{kind} · {source.get('locator', '')}", source.get("url", ""),
        f"Retrieved {retrieved}" if source.get("kind") != "file" else "",
        "Partial coverage" if source.get("partial") else "",
        source.get("text", ""),
    ])))


class Dialog(ModalScreen):
    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    Dialog { align: center middle; background: $background 70%; }
    Dialog > Vertical { width: 74; max-width: 96%; max-height: 94%; height: auto; border: ascii $accent; background: $surface; padding: 1 2; }
    Dialog .dialog-title { text-style: bold; color: $accent; margin-bottom: 1; height: auto; }
    Dialog .dialog-body { height: auto; max-height: 50vh; }
    Dialog Label { height: auto; margin-top: 1; }
    Dialog .help { color: $text-muted; height: auto; }
    Dialog .error { color: $error; height: auto; }
    Dialog .actions { height: auto; min-height: 3; margin-top: 1; }
    Dialog Button { min-width: 10; margin-right: 1; }
    Dialog TextArea { height: 5; min-height: 3; }
    Dialog Input, Dialog Select { height: 3; }
    Dialog TabbedContent { height: 50vh; }
    Dialog TabPane { padding: 0; height: 1fr; }
    Dialog TabPane > VerticalScroll { height: 1fr; max-height: 100%; }
    """

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmDialog(Dialog):
    def __init__(self, title: str, message: str, yes: str = "Continue"):
        super().__init__()
        self.title_text, self.message, self.yes = title, message, yes

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain(self.title_text, classes="dialog-title")
            yield plain(self.message)
            with Horizontal(classes="actions"):
                yield Button(self.yes, variant="primary", id="confirm-yes")
                yield Button("Cancel", id="confirm-no")

    @on(Button.Pressed)
    def answer(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")


class TextDialog(Dialog):
    def __init__(self, title: str, message: str):
        super().__init__()
        self.title_text, self.message = title, message

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain(self.title_text, classes="dialog-title")
            with VerticalScroll(classes="dialog-body"):
                yield plain(self.message)
            yield Button("Close", id="close-text")

    @on(Button.Pressed, "#close-text")
    def close(self) -> None:
        self.dismiss(None)


class ReplyDialog(Dialog):
    DEFAULT_CSS = """
    ReplyDialog > Vertical { width: 110; height: 88%; }
    ReplyDialog #reply-views { height: 1fr; min-height: 4; }
    ReplyDialog TabPane { height: 1fr; padding: 0; }
    ReplyDialog #reply-reader { height: 1fr; }
    ReplyDialog #reply-raw { height: 1fr; min-height: 3; }
    ReplyDialog .actions { height: 3; min-height: 3; }
    """

    def __init__(self, turn: dict):
        super().__init__()
        self.turn = turn

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain("Assistant reply", classes="dialog-title")
            with TabbedContent(id="reply-views"):
                with TabPane("Read", id="reply-read-tab"):
                    with VerticalScroll(id="reply-reader"):
                        yield Static(Markdown(safe(self.turn["answer"]), hyperlinks=False), id="reply-rendered")
                with TabPane("Raw text", id="reply-raw-tab"):
                    yield plain("Raw text scrolls sideways for wide tables and code.", classes="help")
                    yield TextArea(safe(self.turn["answer"]), read_only=True, soft_wrap=False, id="reply-raw")
            with Horizontal(classes="actions"):
                yield Button("Select all", id="reply-select-all")
                yield Button("Copy", id="reply-copy")
                yield Button("Save reply…", id="reply-save", variant="primary")
                yield Button("Close", id="reply-close")

    @on(Button.Pressed)
    def reply_action(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "reply-select-all":
            self.query_one("#reply-views", TabbedContent).active = "reply-raw-tab"
            raw = self.query_one("#reply-raw", TextArea)
            raw.focus()
            raw.action_select_all()
        elif event.button.id == "reply-copy":
            selected = self.query_one("#reply-raw", TextArea).selected_text
            self.app.copy_to_clipboard(selected or safe(self.turn["answer"]))
            self.notify("Copied in this app; terminal clipboard requested.")
        elif event.button.id == "reply-save":
            self.dismiss(self.turn["id"])
        elif event.button.id == "reply-close":
            self.dismiss(None)


class SaveReplyDialog(Dialog):
    def __init__(self, turn_id: str, workspace: str):
        super().__init__()
        self.turn_id, self.workspace = turn_id, workspace

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain("Save reply", classes="dialog-title")
            with VerticalScroll(classes="dialog-body"):
                yield plain(f"Saves only this assistant reply in your HPC workspace:\n{self.workspace}", classes="help")
                yield Label("File path, relative to workspace")
                yield Input(f"reply-{self.turn_id[:8]}.md", id="reply-destination")
                yield Select([("Markdown (.md)", "md"), ("Plain text (.txt)", "txt")], value="md", allow_blank=False, id="reply-format")
                yield Checkbox("Allow overwriting this exact destination", id="reply-overwrite")
            with Horizontal(classes="actions"):
                yield Button("Save", id="save-reply-confirm", variant="primary")
                yield Button("Cancel", id="save-reply-cancel")

    @on(Select.Changed, "#reply-format")
    def format_changed(self, event: Select.Changed) -> None:
        if self.query("#reply-destination"):
            destination = self.query_one("#reply-destination", Input)
            if Path(destination.value).suffix.lower() in {".md", ".txt"}:
                destination.value = str(Path(destination.value).with_suffix("." + str(event.value)))

    @on(Button.Pressed)
    def save_action(self, event: Button.Pressed) -> None:
        event.stop()
        self.dismiss({"turn_id": self.turn_id, "destination": self.query_one("#reply-destination", Input).value,
                      "format": self.query_one("#reply-format", Select).value,
                      "overwrite": self.query_one("#reply-overwrite", Checkbox).value} if event.button.id == "save-reply-confirm" else None)


class ValueDialog(Dialog):
    def __init__(self, title: str, label: str, value: str = "", *, password: bool = False, action_label: str = "Apply"):
        super().__init__()
        self.title_text, self.label_text, self.value, self.password = title, label, value, password
        self.action_label = action_label

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain(self.title_text, classes="dialog-title")
            yield plain(self.label_text)
            yield Input(self.value, id="dialog-value", password=self.password)
            with Horizontal(classes="actions"):
                yield Button(self.action_label, variant="primary", id="value-apply")
                yield Button("Cancel", id="value-cancel")

    @on(Button.Pressed)
    def answer(self, event: Button.Pressed) -> None:
        self.dismiss(self.query_one("#dialog-value", Input).value if event.button.id == "value-apply" else None)


class ThinkingDialog(Dialog):
    def __init__(self, settings: dict, capabilities: dict):
        super().__init__()
        self.settings, self.capabilities = settings, capabilities

    def compose(self) -> ComposeResult:
        supported = self.capabilities.get("thinking") == "enable_thinking"
        default = self.capabilities.get("reasoning_default")
        auto_label = f"Auto (default: {thinking_label(default)})" if default else "Auto (model default)"
        choices = [(auto_label, "auto")]
        if supported:
            choices.extend([("On (model default)", "on"), ("Off", "off")])
        choices.extend((thinking_label(level), level) for level in self.capabilities.get("reasoning_efforts", []) if level in {"minimal", "low", "medium", "high", "xhigh"})
        current = self.settings.get("thinking", "auto")
        if current not in {value for _, value in choices}:
            current = "auto"
        with Vertical():
            yield plain("Thinking", classes="dialog-title")
            yield plain("Auto follows this model's template. Controls below are declared by the loaded template. Effort changes generation; showing reasoning is a separate preference.", classes="help")
            yield Select(choices, value=current, allow_blank=False, id="thinking-value")
            yield Checkbox("Show reasoning by default", value=self.settings.get("show_reasoning", False), id="thinking-show")
            yield plain(self.capabilities.get("provenance", "Capability unknown; only Auto is available."), classes="help")
            with Horizontal(classes="actions"):
                yield Button("Apply", id="thinking-apply", variant="primary")
                yield Button("Cancel", id="thinking-cancel")

    @on(Button.Pressed)
    def answer(self, event: Button.Pressed) -> None:
        self.dismiss({"thinking": self.query_one("#thinking-value", Select).value, "show_reasoning": self.query_one("#thinking-show", Checkbox).value} if event.button.id == "thinking-apply" else None)


def thinking_label(value: str) -> str:
    return "XHigh" if value == "xhigh" else value.title()


SAMPLING = [
    ("temperature", "Temperature", float, "0–2; higher values vary wording more"),
    ("top_p", "Top-p", float, "0–1; cumulative token probability"),
    ("top_k", "Top-k", int, "0–1000; 0 disables the candidate limit"),
    ("min_p", "Min-p", float, "0–1; relative probability cutoff"),
    ("seed", "Seed", int, "-1 chooses a random seed"),
    ("repeat_penalty", "Repeat penalty", float, "1 leaves repetition unchanged"),
    ("presence_penalty", "Presence penalty", float, "-2 to 2"),
    ("frequency_penalty", "Frequency penalty", float, "-2 to 2"),
]
RUNTIME = [
    ("gpu_layers", "GPU layers", int, "-1 requests all layers; actual offload is checked in diagnostics"),
    ("threads", "Generation CPU threads", int, "Keep within the allocated CPU budget"),
    ("threads_batch", "Prompt CPU threads", int, "Keep within the allocated CPU budget"),
    ("batch_size", "Batch size", int, "Tokens per prompt batch"),
    ("ubatch_size", "Microbatch size", int, "Must not exceed batch size"),
]
ACCELERATION = [
    ("spec_draft_n_max", "Maximum draft tokens", int, "1–64; larger drafts need more memory and may be slower"),
    ("spec_draft_n_min", "Minimum draft tokens", int, "0–64; must not exceed the maximum"),
    ("spec_draft_p_min", "Draft probability threshold", float, "0–1; reject uncertain draft tokens"),
]
PRIVACY = [
    ("history_messages", "Previous messages to include", int, "Blank includes full history; 0 explicitly excludes prior messages"),
    ("tool_limit", "Tool actions per turn", int, "1–12; no arbitrary code execution"),
    ("tool_timeout", "Tool timeout, seconds", int, "5–300"),
    ("approval_timeout", "Approval expiry, seconds", int, "10–1800; detach never grants consent"),
    ("idle_shutdown_minutes", "Idle shutdown, minutes", int, "0 disables automatic idle shutdown"),
]
RESOURCE_FIELDS = [
    ("partition", "Partition", str), ("gpu_type", "GPU type", str),
    ("gpu_count", "GPU count", int), ("cpus", "CPU count", int),
    ("memory_gb", "Host memory, GB", int), ("walltime", "Duration HH:MM:SS", str),
    ("account", "Account (optional)", str), ("qos", "QOS (optional)", str),
    ("constraint", "Constraint (optional)", str),
]


class AdvancedDialog(Dialog):
    DEFAULT_CSS = """
    AdvancedDialog #acceleration-status { width: 100%; height: auto; text-wrap: wrap; }
    """

    def __init__(self, settings: dict, capabilities: dict, resources: dict):
        super().__init__()
        self.settings, self.capabilities, self.resources = dict(settings), capabilities, resources
        self.defaults = InferenceSettings().model_dump()

    def fields(self, definitions: list, supported: list | None = None) -> ComposeResult:
        for name, title, kind, hint in definitions:
            disabled = supported is not None and name not in supported
            yield Label(f"{title}  ·  default {self.defaults.get(name)}")
            yield Input("" if self.settings.get(name) is None else str(self.settings[name]), id=f"adv-{name}", disabled=disabled)
            yield plain("Unsupported by this runtime" if disabled else hint, classes="help")

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain("Advanced settings", classes="dialog-title")
            with TabbedContent():
                with TabPane("Sampling", id="sampling-tab"):
                    with VerticalScroll(classes="dialog-body"):
                        yield plain("Applies to the next message. Does not restart the model.", classes="help")
                        yield from self.fields(SAMPLING, self.capabilities.get("sampling", []))
                with TabPane("Runtime", id="runtime-tab"):
                    with VerticalScroll(classes="dialog-body"):
                        yield plain("Changes reload the model inside this allocation after confirmation. One inference slot.", classes="help")
                        yield Label("Acceleration")
                        yield Select([("Off", "off"), ("Auto · use MTP when available", "auto"), ("MTP · require support", "mtp")], value=self.settings.get("acceleration", "off"), allow_blank=False, id="adv-acceleration")
                        yield plain(f"{self.capabilities.get('acceleration_status', 'Off')}\n{self.capabilities.get('acceleration_reason', 'MTP is disabled')}", id="acceleration-status", classes="help")
                        yield plain("Auto falls back to ordinary generation when MTP is unavailable. MTP requires a compatible runtime and embedded layers or a matching head. Adding companions in Models takes effect in a new allocation.", classes="help")
                        with Collapsible(title="MTP draft tuning", collapsed=True):
                            yield from self.fields(ACCELERATION, self.capabilities.get("runtime_controls", []))
                        controls = self.capabilities.get("runtime_controls", [])
                        yield from self.fields(RUNTIME, controls)
                        for name, title in [("cache_type_k", "Key cache"), ("cache_type_v", "Value cache")]:
                            yield Label(title)
                            yield Select([(v, v) for v in ["f16", "q8_0", "q4_0"]], value=self.settings[name], allow_blank=False, id=f"adv-{name}", disabled=name not in controls)
                        yield Label("Flash attention")
                        yield Select([(v.title(), v) for v in ["auto", "on", "off"]], value=self.settings["flash_attention"], allow_blank=False, id="adv-flash_attention", disabled="flash_attention" not in controls)
                with TabPane("Tools / privacy", id="privacy-tab"):
                    with VerticalScroll(classes="dialog-body"):
                        yield plain("Web On permits external queries and page requests until Web Off. File processing and inference stay local. No automatic provider fallback.", classes="help")
                        yield Checkbox("Ask before each external web request", value=self.settings.get("web_approval") == "per_request", id="adv-web-approval")
                        yield from self.fields(PRIVACY)
                with TabPane("Resources", id="resources-tab"):
                    with VerticalScroll(classes="dialog-body"):
                        yield plain("Requires a new allocation. Current resources cannot be resized here.", classes="help")
                        yield plain("\n".join(f"{title}: {self.resources.get(name) or '—'}" for name, title, _ in RESOURCE_FIELDS))
            yield plain("", id="advanced-error", classes="error")
            with Horizontal(classes="actions"):
                yield Button("Reset", id="advanced-reset")
                yield Button("Cancel", id="advanced-cancel")
                yield Button("Apply", id="advanced-apply", variant="primary")

    def collect(self) -> dict:
        result = dict(self.settings)
        for name, _, kind, _ in SAMPLING + RUNTIME + ACCELERATION + PRIVACY:
            field = self.query_one(f"#adv-{name}", Input)
            if not field.disabled:
                result[name] = None if name == "history_messages" and not field.value.strip() else kind(field.value)
        for name in ["cache_type_k", "cache_type_v", "flash_attention", "acceleration"]:
            field = self.query_one(f"#adv-{name}", Select)
            if not field.disabled:
                result[name] = field.value
        result["web_approval"] = "per_request" if self.query_one("#adv-web-approval", Checkbox).value else "session"
        return InferenceSettings(**result).model_dump()

    @on(Button.Pressed)
    def buttons(self, event: Button.Pressed) -> None:
        if event.button.id == "advanced-cancel":
            self.dismiss(None)
        elif event.button.id == "advanced-reset":
            for name, _, _, _ in SAMPLING + RUNTIME + ACCELERATION + PRIVACY:
                field = self.query_one(f"#adv-{name}", Input)
                if not field.disabled:
                    field.value = "" if self.defaults[name] is None else str(self.defaults[name])
            for name in ["cache_type_k", "cache_type_v", "flash_attention", "acceleration"]:
                self.query_one(f"#adv-{name}", Select).value = self.defaults[name]
            self.query_one("#adv-web-approval", Checkbox).value = False
        elif event.button.id == "advanced-apply":
            try:
                self.dismiss(self.collect())
            except (ValueError, TypeError) as exc:
                self.query_one("#advanced-error", Static).update(safe(exc))


class SettingsDialog(Dialog):
    def __init__(self, manifest: dict, *, focus_context: bool = False):
        super().__init__()
        self.manifest = manifest
        self.pending = dict(manifest["settings"])
        self.focus_context = focus_context

    def compose(self) -> ComposeResult:
        caps = self.manifest.get("capabilities", {})
        with Vertical():
            yield plain("Settings", classes="dialog-title")
            with VerticalScroll(classes="dialog-body"):
                yield plain(f"Model: {self.manifest['model']['name']}")
                yield plain("Model and resources require a new session; current GPU continues until explicitly stopped.", classes="help")
                yield Button("New allocation…", id="settings-allocation")
                yield Label("Context capacity (reloads model)")
                yield plain(f"Loaded: {caps.get('loaded_context', '?'):,} tokens  ·  Model limit: {caps.get('supported_context') or 'unknown'}", id="context-effective", classes="help")
                yield Select([("4k", 4096), ("8k", 8192), ("16k", 16384), ("32k", 32768), ("Custom", 0)], value=self.pending["context"] if self.pending["context"] in [4096, 8192, 16384, 32768] else 0, allow_blank=False, id="context-preset")
                yield Input(str(self.pending["context"]), id="setting-context", type="integer")
                yield plain("", id="context-vram", classes="help")
                yield Label("Maximum response tokens (0 = Auto)")
                yield Input(str(self.pending["max_tokens"]), id="setting-max_tokens", type="integer")
                yield plain("Auto uses the estimated remaining context for the answer and thinking. A positive limit is capped to the space available.", classes="help")
                yield Label("System instructions")
                yield TextArea(self.pending["system_prompt"], id="setting-system_prompt")
                yield Checkbox("Web enabled: allow external searches and page requests", value=self.pending["web_enabled"], id="setting-web_enabled")
                yield Label("Search provider")
                options = [("No-key search · best effort", "ddgs"), ("Brave · own API key", "brave")]
                if self.manifest.get("demo"):
                    options.append(("Controlled demo fixtures", "fixture"))
                yield Select(options, value=self.pending["web_provider"], allow_blank=False, id="setting-web_provider")
                yield Button("Provider status / key…", id="settings-key")
                yield Button("Advanced…", id="settings-advanced")
                yield Checkbox("Save these settings as this model's default", id="settings-default")
            yield plain("", id="settings-error", classes="error")
            with Horizontal(classes="actions"):
                yield Button("Reset", id="settings-reset")
                yield Button("Cancel", id="settings-cancel")
                yield Button("Apply", id="settings-apply", variant="primary")

    def on_mount(self) -> None:
        self.update_vram()
        if self.focus_context:
            self.query_one("#setting-context").focus()

    @on(Input.Changed, "#setting-context")
    def context_changed(self, event: Input.Changed) -> None:
        self.update_vram()

    def update_vram(self) -> None:
        if not self.is_mounted:
            return
        try:
            context = int(self.query_one("#setting-context", Input).value)
        except ValueError:
            text = "VRAM estimate: enter a valid context size."
        else:
            text = estimate_vram(self.manifest["model"], dict(self.pending, context=context), self.manifest.get("capabilities", {}), self.manifest["resources"])
        if text != getattr(self, "_vram_text", None):
            self._vram_text = text
            self.query_one("#context-vram", Static).update(safe(text))

    @on(Select.Changed, "#context-preset")
    def preset(self, event: Select.Changed) -> None:
        if isinstance(event.value, int) and event.value:
            self.query_one("#setting-context", Input).value = str(event.value)

    def collect(self) -> dict:
        result = dict(self.pending)
        for name in ["context", "max_tokens"]:
            result[name] = int(self.query_one(f"#setting-{name}", Input).value)
        result["system_prompt"] = self.query_one("#setting-system_prompt", TextArea).text
        result["web_enabled"] = self.query_one("#setting-web_enabled", Checkbox).value
        result["web_provider"] = self.query_one("#setting-web_provider", Select).value
        return InferenceSettings(**result).model_dump()

    @on(Button.Pressed)
    def buttons(self, event: Button.Pressed) -> None:
        id_ = event.button.id
        if id_ == "settings-cancel":
            self.dismiss(None)
        elif id_ == "settings-allocation":
            self.dismiss({"new_allocation": True})
        elif id_ == "settings-key":
            self.app.show_provider()
        elif id_ == "settings-reset":
            defaults = InferenceSettings().model_dump()
            self.pending = defaults
            self.query_one("#setting-context", Input).value = str(defaults["context"])
            self.query_one("#setting-max_tokens", Input).value = str(defaults["max_tokens"])
            self.query_one("#setting-system_prompt", TextArea).load_text(defaults["system_prompt"])
            self.query_one("#setting-web_enabled", Checkbox).value = False
            self.query_one("#setting-web_provider", Select).value = "fixture" if self.manifest.get("demo") else "ddgs"
            self.update_vram()
        elif id_ in {"settings-apply", "settings-advanced"}:
            try:
                self.pending = self.collect()
            except (ValueError, TypeError) as exc:
                self.query_one("#settings-error", Static).update(safe(exc))
                return
            if id_ == "settings-advanced":
                self.app.push_screen(AdvancedDialog(self.pending, self.manifest.get("capabilities", {}), self.manifest["resources"]), self.receive_advanced)
            else:
                self.dismiss({"settings": self.pending, "save_model_default": self.query_one("#settings-default", Checkbox).value})

    def receive_advanced(self, settings: dict | None) -> None:
        if settings:
            self.pending = settings
            self.update_vram()


class ApprovalDialog(Dialog):
    def __init__(self, approval: dict):
        super().__init__()
        self.approval = approval
        self.key = "query" if approval.get("name") == "web_search" else "url"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain("Review external request", classes="dialog-title")
            yield plain(f"Action: {self.approval['name']}  ·  Provider: {self.approval.get('provider') or 'public website'}")
            yield plain("Only this exact query or URL is approved. It may contain sensitive terms. Inference remains local; external services receive this request.", classes="help")
            yield Input(str(self.approval.get("arguments", {}).get(self.key, "")), id="approval-value")
            yield plain("Editing replaces the pending request with a newly bound approval.", classes="help")
            with Horizontal(classes="actions"):
                yield Button("Approve", variant="primary", id="approval-yes")
                yield Button("Deny", id="approval-no")

    @on(Button.Pressed)
    def answer(self, event: Button.Pressed) -> None:
        args = dict(self.approval["arguments"])
        args[self.key] = self.query_one("#approval-value", Input).value
        self.dismiss({"approve": event.button.id == "approval-yes", "arguments": args})


class SourceSelectionDialog(Dialog):
    def __init__(self, sources: list[dict]):
        super().__init__()
        self.sources = sources

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain("Choose evidence for the next answer", classes="dialog-title")
            with VerticalScroll(classes="dialog-body"):
                for i, source in enumerate(self.sources):
                    yield Checkbox(safe(f"[{source['id']}] {source['title']}"), value=True, id=f"source-pick-{i}")
                    with Collapsible(title="Inspect excerpt", collapsed=True):
                        yield plain(source_text(source))
                    if source.get("url"):
                        yield Button(f"Read page {source['id']}", id=f"source-fetch-{i}")
            with Horizontal(classes="actions"):
                yield Button("Use selected", id="sources-use", variant="primary")
                yield Button("Close", id="sources-close")

    @on(Button.Pressed)
    def buttons(self, event: Button.Pressed) -> None:
        id_ = event.button.id or ""
        if id_.startswith("source-fetch-"):
            source = self.sources[int(id_.rsplit("-", 1)[-1])]
            self.dismiss({"fetch": source["url"]})
        elif id_ == "sources-use":
            self.dismiss({"source_ids": [s["id"] for i, s in enumerate(self.sources) if self.query_one(f"#source-pick-{i}", Checkbox).value]})
        elif id_ == "sources-close":
            self.dismiss(None)


class AttachDialog(Dialog):
    def __init__(self, path: str = ""):
        super().__init__()
        self.path_text = path

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain("Attach local file", classes="dialog-title")
            with VerticalScroll(classes="dialog-body"):
                yield Label("Path inside the chosen workspace")
                yield Input(self.path_text, id="attach-path")
                yield plain("Files are parsed on the allocated node. Large files use bounded excerpts with visible coverage.", classes="help")
                yield plain("Upload laptop files through Open OnDemand Files or SFTP first; this browser shows files already on HPC.", classes="help")
                yield Label("Pages (PDF, optional; e.g. 1-3)")
                yield Input(placeholder="All within parser limits", id="attach-pages")
                yield Label("Lines (text, optional; e.g. 1-100)")
                yield Input(placeholder="All within parser limits", id="attach-lines")
                yield Label("Sheet (XLSX, optional)")
                yield Input(id="attach-sheet")
                yield Label("Cells (spreadsheet, optional; e.g. A1:D20)")
                yield Input(id="attach-cells")
                with Collapsible(title="CSV / JSON selection", collapsed=True):
                    yield Label("CSV first data row / last row (optional; header is row 1)")
                    yield Input(id="attach-start_row", type="integer", placeholder="First row, e.g. 2")
                    yield Input(id="attach-end_row", type="integer", placeholder="Last row, e.g. 100")
                    yield Label("CSV columns (comma-separated names, optional)")
                    yield Input(id="attach-columns", placeholder="treatment,value")
                    yield Label("JSON pointer (optional)")
                    yield Input(id="attach-pointer", placeholder="/results/0")
            yield plain("", id="attach-error", classes="error")
            with Horizontal(classes="actions"):
                yield Button("Attach / preview", variant="primary", id="attach-apply")
                yield Button("Cancel", id="attach-cancel")

    @on(Button.Pressed)
    def buttons(self, event: Button.Pressed) -> None:
        if event.button.id == "attach-cancel":
            self.dismiss(None)
        elif event.button.id == "attach-apply":
            path = self.query_one("#attach-path", Input).value.strip()
            if not path:
                self.query_one("#attach-error", Static).update("Choose a file path.")
                return
            selection = {("range" if k == "cells" else k): self.query_one(f"#attach-{k}", Input).value.strip() for k in ["pages", "lines", "sheet", "cells"] if self.query_one(f"#attach-{k}", Input).value.strip()}
            try:
                for key in ["start_row", "end_row"]:
                    value = self.query_one(f"#attach-{key}", Input).value.strip()
                    if value:
                        selection[key] = int(value)
                columns = self.query_one("#attach-columns", Input).value.strip()
                if columns:
                    selection["columns"] = [value.strip() for value in columns.split(",") if value.strip()]
                pointer = self.query_one("#attach-pointer", Input).value.strip()
                if pointer:
                    selection["pointer"] = pointer
            except ValueError:
                self.query_one("#attach-error", Static).update("CSV row bounds must be whole numbers.")
                return
            self.dismiss({"path": path, "selection": selection})


class AttachmentsDialog(Dialog):
    def __init__(self, attachments: list[dict], selected: list[str]):
        super().__init__()
        self.attachments, self.selected = attachments, selected

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain("Files for this chat", classes="dialog-title")
            yield plain("Uncheck a file to leave it out of the next message. Saved source snapshots stay in chat history.", classes="help")
            with VerticalScroll(classes="dialog-body"):
                if not self.attachments:
                    yield plain("No files attached yet. Choose Attach from the Files panel.")
                for i, attachment in enumerate(self.attachments):
                    yield Checkbox(safe(f"{attachment['name']} · {attachment['status']}"), value=attachment["id"] in self.selected, id=f"attachment-pick-{i}", disabled=attachment["status"] in {"error", "unsupported"})
                    with Collapsible(title="Preview and coverage", collapsed=True):
                        yield plain(f"Type: {attachment.get('media_type', 'unknown')} · Size: {attachment.get('size_bytes', 0):,} bytes\nSelection: {attachment.get('selection') or 'default bounded extraction'}")
                        yield plain("\n".join(attachment.get("warnings", [])))
                        yield plain(f"Showing {min(12, len(attachment.get('sources', [])))} of {len(attachment.get('sources', []))} extracted source records; selected content remains available for retrieval.")
                        for source in attachment.get("sources", [])[:12]:
                            yield plain(source_text(source))
            with Horizontal(classes="actions"):
                yield Button("Use selected", id="attachments-use", variant="primary")
                yield Button("Cancel", id="attachments-cancel")

    @on(Button.Pressed)
    def buttons(self, event: Button.Pressed) -> None:
        self.dismiss([a["id"] for i, a in enumerate(self.attachments) if self.query_one(f"#attachment-pick-{i}", Checkbox).value] if event.button.id == "attachments-use" else None)


class ExportDialog(Dialog):
    def __init__(self, default_path: str):
        super().__init__()
        self.default_path = default_path

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain("Save result", classes="dialog-title")
            yield Label("Destination in workspace (originals are never changed automatically)")
            yield Input(self.default_path, id="export-path")
            yield Select([("Markdown", "md"), ("Plain text", "txt"), ("JSON", "json"), ("CSV (formula-safe)", "csv")], value="md", allow_blank=False, id="export-format")
            yield Checkbox("Allow overwriting this exact destination", id="export-overwrite")
            yield plain("CSV exports neutralize spreadsheet formulas from untrusted text.", classes="help")
            with Horizontal(classes="actions"):
                yield Button("Save", id="export-save", variant="primary")
                yield Button("Cancel", id="export-cancel")

    @on(Button.Pressed)
    def answer(self, event: Button.Pressed) -> None:
        self.dismiss({"destination": self.query_one("#export-path", Input).value, "format": self.query_one("#export-format", Select).value, "overwrite": self.query_one("#export-overwrite", Checkbox).value} if event.button.id == "export-save" else None)


class ChatApp(App):
    """Attachable, API-only terminal application. ``client`` is SessionClient."""
    TITLE = "HPC LLM"
    ENABLE_COMMAND_PALETTE = False
    DETACH_SAVE_TIMEOUT = 5.0
    BINDINGS = [
        Binding("ctrl+s", "send", "Send / stop", priority=True),
        Binding("ctrl+b", "panel", "Files / chats", priority=True, show=False),
        Binding("ctrl+n", "new_chat", "New chat", priority=True, show=False),
        Binding("ctrl+o", "attach", "Attach", priority=True),
        Binding("ctrl+p", "settings", "Settings", priority=True),
        Binding("ctrl+d", "detach", "Detach", priority=True),
        Binding("ctrl+q", "quit", "Detach", priority=True, show=False),
        Binding("f1", "help", "Help", priority=True),
    ]
    CSS = """
    Screen { background: $background; }
    #session-status { height: 1; padding: 0 1; background: $boost; color: $text; text-style: bold; }
    #toolbar { height: 3; padding: 0 1; }
    #toolbar Button { min-width: 7; height: 3; padding: 0; margin-right: 0; }
    #body { height: 1fr; }
    #drawer { width: 29; border-right: ascii $primary-background; padding: 0 1; }
    #drawer TabbedContent { height: 1fr; }
    #drawer TabPane { padding: 0; }
    #drawer Button { width: 100%; min-width: 8; margin-bottom: 0; }
    #file-tree { height: 1fr; min-height: 3; }
    #file-filter { height: 3; }
    #chat-list { height: 1fr; }
    #transcript { height: 1fr; padding: 0 2; }
    .turn { height: auto; margin: 1 0; }
    .role { color: $accent; text-style: bold; height: auto; }
    .answer, .question { height: auto; }
    .reply-heading { height: 1; }
    .reply-heading .role { width: 1fr; }
    .reply-action, .output-copy { width: auto; min-width: 10; height: 1; min-height: 1; padding: 0 1; border: none !important; }
    #generated-outputs, .output-card { height: auto; }
    .output-card { margin: 1 0; padding-left: 1; border-left: ascii $accent; }
    .turn-meta { height: auto; color: $text-muted; }
    #empty { height: auto; margin-top: 2; color: $text-muted; }
    #activity-row { height: auto; min-height: 1; max-height: 2; padding: 0 1; }
    #activity { width: 1fr; height: auto; max-height: 2; color: $text-muted; }
    #new-chat-main, #copy-reply, #paste-copied { width: auto; min-width: 10; height: 1; min-height: 1; padding: 0 1; border: none !important; }
    #tray { height: auto; min-height: 0; padding: 0 1; }
    #tray Button { height: 1; min-height: 1; min-width: 8; border: none; padding: 0 1; margin-right: 1; }
    #composer-row { height: 5; padding: 0 1; }
    #composer { width: 1fr; height: 5; border: ascii $primary; }
    #send { width: 10; min-width: 8; height: 5; margin-left: 1; }
    Footer { height: 1; }
    .error { color: $error; }
    """ + PORTABLE_BORDERS_CSS

    def __init__(self, client):
        super().__init__()
        self.client = client
        self.snapshot: dict = {}
        self.conversation_id: str | None = None
        self.attachment_ids: list[str] = []
        self.source_ids: list[str] = []
        self._polling = False
        self._settings_operation = ""
        self._render_key = ""
        self._transcript_structure = ""
        self._turn_render_keys: dict[str, str] = {}
        self._output_render_key = ""
        self._chat_list_key = ""
        self._draft_timer = None
        self._refresh_timer = None
        self._draft_lock = asyncio.Lock()
        self._detach_prompt = False
        self._loading_draft = False
        self._draft_dirty = False
        self._last_sent_draft = ""
        self._seen_approvals: set[str] = set()
        self._seen_results: set[str] = set()
        self._panel_choice: bool | None = None
        self._file_root = ""
        self._tree_loaded: set[str] = set()
        self._detaching = False
        self.last_error = ""

    @property
    def manifest(self) -> dict:
        return self.snapshot.get("manifest", data(getattr(self.client, "manifest", {})))

    @property
    def conversation(self) -> dict:
        return next((c for c in records(self.snapshot.get("conversations")) if c["id"] == self.conversation_id), {})

    def compose(self) -> ComposeResult:
        yield plain("HPC LLM · Connecting…", id="session-status")
        with Horizontal(id="toolbar"):
            yield Button("Files/Chats", id="panel")
            yield Button("Think: Auto", id="thinking")
            yield Button("Context", id="context")
            yield Button("Web: Off", id="web")
            yield Button("Search", id="search")
            yield Button("Settings", id="settings")
        with Horizontal(id="body"):
            with Vertical(id="drawer"):
                with TabbedContent():
                    with TabPane("Files", id="files-tab"):
                        yield Input(placeholder="Find file…", id="file-filter")
                        yield Tree("Workspace", id="file-tree")
                        yield Button("Attach…", id="attach")
                        yield Button("Selected files…", id="attachments")
                        yield Button("Choose workspace…", id="workspace")
                        yield Button("File actions…", id="file-actions")
                    with TabPane("Chats", id="chats-tab"):
                        yield Button("New chat", id="new-chat")
                        with VerticalScroll(id="chat-list"):
                            yield plain("Loading saved chats…")
                        yield Button("Retry last message", id="retry")
                        yield Button("Save result…", id="export")
                        yield Button("Delete chat…", id="delete-chat")
                        yield Button("Detach", id="detach")
                        yield Button("Stop GPU session…", id="stop-session")
            with Vertical():
                with VerticalScroll(id="transcript"):
                    yield plain("Connecting to your saved session…", id="empty")
                with Horizontal(id="activity-row"):
                    yield plain("", id="activity")
                    yield Button("Copy reply", id="copy-reply", compact=True, disabled=True, tooltip="Copy the latest assistant reply. Requests the terminal clipboard and keeps an in-app copy.")
                    yield Button("Paste copied", id="paste-copied", compact=True, tooltip="Paste text copied inside this app. For outside text, use your terminal's paste command.")
                    yield Button("New chat", id="new-chat-main", compact=True, tooltip="Start an empty conversation (Ctrl+N). This chat stays saved.")
                with Horizontal(id="tray"):
                    yield Button("", id="tray-files")
                    yield Button("", id="tray-sources")
                with Horizontal(id="composer-row"):
                    yield ChatComposer(placeholder="Message… Enter sends; Shift+Enter / Ctrl+J adds a line", id="composer")
                    yield Button("Send", id="send", variant="primary")
        yield Footer()

    async def on_mount(self) -> None:
        self.theme = "textual-dark"
        self.query_one("#tray").display = False
        self.query_one("#paste-copied").display = False
        self.query_one("#drawer").display = self.size.width >= 108
        await self.refresh_state(initial=True)
        self._refresh_timer = self.set_interval(0.4, self.refresh_state)
        self.query_one("#composer", TextArea).focus()

    async def api(self, method: str, path: str, **kwargs: Any) -> Any:
        return await self.client.request(method, path, **kwargs)

    def report_error(self, exc: Exception | str) -> None:
        self.last_error = safe(getattr(exc, "message", str(exc)))
        if self.screen_stack and self.screen_stack[0].query("#activity"):
            self.screen_stack[0].query_one("#activity", Static).update(self.last_error)
        self.notify(self.last_error, title="Action not completed", severity="error", timeout=8)

    def accept_snapshot(self, snapshot: dict) -> None:
        """Do not let an older in-flight poll undo a confirmed settings response."""
        current = self.snapshot.get("manifest", {})
        incoming = snapshot.get("manifest", {})
        if current.get("settings_revision", 0) > incoming.get("settings_revision", 0):
            snapshot = dict(snapshot, manifest=current)
        self.snapshot = snapshot

    async def refresh_state(self, initial: bool = False) -> None:
        if self._polling or self._detaching:
            return
        self._polling = True
        try:
            snapshot = await self.client.state()
            if self._detaching:
                return
            self.accept_snapshot(snapshot)
            for dialog in self.screen_stack:
                if isinstance(dialog, SettingsDialog):
                    incoming = self.snapshot.get("manifest", {})
                    if incoming.get("id") == dialog.manifest.get("id"):
                        dialog.manifest = incoming
                        dialog.update_vram()
            if self._detaching or not self.query("#session-status"):
                return
            if not self._file_root:
                self._file_root = self.snapshot.get("workspace", "")
            conversations = records(self.snapshot.get("conversations"))
            if not self.conversation_id or not any(c["id"] == self.conversation_id for c in conversations):
                if conversations:
                    await self.open_conversation(conversations[-1]["id"], flush=False)
                else:
                    created = await self.api("POST", "/conversations", json={})
                    self.conversation_id = created["id"]
                    self.snapshot["conversations"] = [created]
                    self.load_draft(created)
            self.update_header()
            await self.render_transcript()
            await self.render_chat_list()
            self.update_tray()
            if initial:
                await self.load_files()
            await self.check_approvals()
        except Exception as exc:
            if not self._detaching and self.query("#session-status"):
                self.query_one("#session-status", Static).update("Disconnected · Reconnecting automatically; Ctrl+D detaches")
                if initial:
                    self.report_error(exc)
        finally:
            self._polling = False

    def update_header(self) -> None:
        if not self.screen_stack:
            return
        screen = self.screen_stack[0]
        if not screen.query("#session-status"):
            return
        manifest = self.manifest
        settings, caps = manifest.get("settings", {}), manifest.get("capabilities", {})
        remaining = ""
        if manifest.get("expires_at"):
            remaining = f" · {max(0, int((manifest['expires_at'] - time.time()) / 60))}m remaining"
        mode = "DEMO · " if manifest.get("demo") else ""
        status = self._settings_operation or manifest.get('backend_state', 'connecting').replace('_', ' ')
        screen.query_one("#session-status", Static).update(safe(f"{mode}{manifest.get('model', {}).get('name', 'HPC LLM')} · {status}{remaining}"))
        thinking_button = screen.query_one("#thinking", Button)
        thinking_button.label = f"Think: {thinking_label(settings.get('thinking', 'auto'))}"
        default = caps.get("reasoning_default")
        thinking_button.tooltip = f"Auto uses {thinking_label(default)} for this model. Controls come from the loaded template." if default else "Choose a thinking control declared by the loaded template."
        screen.query_one("#context", Button).label = f"Context {caps.get('loaded_context', settings.get('context', 0)) // 1024}k"
        screen.query_one("#web", Button).label = f"Web: {'On' if settings.get('web_enabled') else 'Off'}"
        for selector in ["#thinking", "#context", "#web", "#settings"]:
            screen.query_one(selector, Button).disabled = bool(self._settings_operation)
        active = self.snapshot.get("active_turn_id")
        screen.query_one("#send", Button).label = "Stop" if active else "Send"
        screen.query_one("#send", Button).variant = "error" if active else "primary"
        screen.query_one("#send", Button).disabled = bool(self._settings_operation and not active)
        screen.query_one("#copy-reply", Button).disabled = not bool(self.last_reply())
        screen.query_one("#paste-copied").display = bool(self.clipboard)
        if not self.last_error:
            text = self._settings_operation or ("Waiting for approval…" if manifest.get("backend_state") == "awaiting_approval" else ("Thinking / responding…" if active else ""))
            turns = self.conversation.get("turns", [])
            if turns and not active and not self._settings_operation:
                used = turns[-1].get("prompt_tokens_estimate", 0)
                if used:
                    limit = turns[-1].get("settings", {}).get("max_tokens", 0)
                    text = f"Last prompt ≈{used:,} · response limit {limit:,} · loaded {caps.get('loaded_context', 0):,}"
            screen.query_one("#activity", Static).update(text)

    async def render_transcript(self) -> None:
        conversation = self.conversation
        turns = conversation.get("turns", [])
        table_results = conversation.get("tool_results", [])
        outputs = conversation.get("outputs", [])
        details = [event for event in self.snapshot.get("events", []) if event.get("kind") in {"warning", "tool", "approval", "status"}]
        key = json.dumps([conversation.get("id"), turns, table_results, outputs, details, self.manifest.get("settings", {}).get("show_reasoning")], sort_keys=True)
        if key == self._render_key:
            return
        self._render_key = key
        container = self.query_one("#transcript", VerticalScroll)
        previous_y = container.scroll_y
        at_end = container.scroll_y >= container.max_scroll_y - 2
        collapsed = {w.id: w.collapsed for w in container.query(Collapsible) if w.id}
        structure = json.dumps([conversation.get("id"), [turn["id"] for turn in turns], table_results, bool(turns or table_results or outputs)], sort_keys=True)
        rebuild = structure != self._transcript_structure
        if rebuild:
            self._transcript_structure = structure
            self._turn_render_keys.clear()
            self._output_render_key = ""
            await container.remove_children()
        if not turns and not table_results and not outputs:
            if rebuild:
                await container.mount(plain("Your model is ready.\n\nWrite a message, attach a local file, or turn on Web to review a search.\nEnter sends. Shift+Enter or Ctrl+J adds a line.\nPaste outside text with your terminal's paste command. F1: help.\n\nChats save automatically. Detach keeps the GPU session running.", id="empty"))
            return
        for turn in turns:
            tid = turn["id"]
            turn_details = [event for event in details if event.get("turn_id") == tid]
            turn_key = json.dumps([turn, turn_details, self.manifest.get("settings", {}).get("show_reasoning")], sort_keys=True)
            if self._turn_render_keys.get(tid) == turn_key:
                continue
            self._turn_render_keys[tid] = turn_key
            answer = turn.get("answer", "")
            if not answer and turn.get("status") in {"running", "awaiting_approval"}:
                answer = "Thinking…" if turn.get("reasoning") else "Working…"
            if not answer and turn.get("finish_reason") == "length":
                answer = "The response budget was used before a final answer. Increase maximum response tokens or turn thinking off where supported, then choose Retry."
            if not answer and turn.get("status") in {"cancelled", "interrupted", "error"}:
                answer = f"Response {turn['status']}. You can retry explicitly."
            tools = [event for event in turn_details if event.get("kind") in {"tool", "approval", "status"}]
            fields = {
                "question": turn.get("request", {}).get("text", ""),
                "answer": answer,
                "reasoning": turn.get("reasoning", ""),
                "sources": "\n\n".join(source_text(source) for source in turn.get("sources", [])),
                "error": turn.get("error") or "",
                "warnings": "\n".join(event.get("data", {}).get("message", "Source coverage warning") for event in turn_details if event.get("kind") == "warning" and event.get("data", {}).get("message") != turn.get("error")),
                "tools": "\n\n".join(json.dumps(event.get("data", {}), indent=2, ensure_ascii=False) for event in tools),
                "limit": "Response stopped at the token budget; use Retry after adjusting maximum response tokens." if turn.get("finish_reason") == "length" and turn.get("answer") else "",
                "status": turn["status"].title() if turn.get("status") in {"cancelled", "interrupted"} else "",
            }
            if rebuild:
                reply_action = Button("Read / save…", id=f"open-reply-{tid}", classes="reply-action", compact=True)
                children = [plain("You", classes="role"), plain(fields["question"], classes="question", id=f"field-question-{tid}"), Horizontal(plain("Assistant", classes="role"), reply_action, classes="reply-heading"), plain(answer, classes="answer", id=f"field-answer-{tid}")]
                for name in ["reasoning", "sources", "error", "warnings", "tools", "limit", "status"]:
                    content = plain(fields[name], classes="error" if name == "error" else "turn-meta" if name in {"warnings", "limit", "status"} else "", id=f"field-{name}-{tid}")
                    if name in {"reasoning", "sources", "tools"}:
                        cid = f"{name}-{tid}"
                        title = {"reasoning": "Reasoning", "sources": "Sources", "tools": "Tool activity"}[name]
                        section = Collapsible(content, title=title, id=cid, collapsed=collapsed.get(cid, not self.manifest["settings"].get("show_reasoning", False) if name == "reasoning" else True))
                    else:
                        section = content
                    section.display = bool(fields[name])
                    children.append(section)
                node = Vertical(*children, classes="turn", id=f"turn-{tid}")
                await container.mount(node)
            else:
                node = container.query_one(f"#turn-{tid}", Vertical)
            # Keep the answer, reasoning and disclosure widgets mounted while
            # streaming. Replacing children every poll causes visible flashing.
            for name, text in fields.items():
                content = node.query_one(f"#field-{name}-{tid}", Static)
                text = safe(text)
                if name == "answer" and turn.get("status") == "completed" and turn.get("answer"):
                    content.update(Markdown(text, hyperlinks=False))
                elif content.content != text:
                    content.update(text)
                if name in {"question", "answer"}:
                    continue
                section = node.query_one(f"#{name}-{tid}", Collapsible) if name in {"reasoning", "sources", "tools"} else content
                if section.display != bool(text):
                    section.display = bool(text)
            source_section = node.query_one(f"#sources-{tid}", Collapsible)
            source_section.title = f"Sources ({len(turn.get('sources', []))}) · inspect citations"
            node.query_one(f"#open-reply-{tid}").display = bool(turn.get("answer") and turn.get("status") == "completed")
        if rebuild:
            for i, result in enumerate(table_results):
                cid = f"table-result-{i}"
                await container.mount(Collapsible(plain(json.dumps(result, indent=2, ensure_ascii=False)), title="Saved deterministic table result", id=cid, collapsed=collapsed.get(cid, True)))
            await container.mount(Vertical(id="generated-outputs"))
        output_key = json.dumps(outputs, sort_keys=True)
        if output_key != self._output_render_key:
            self._output_render_key = output_key
            output_list = container.query_one("#generated-outputs", Vertical)
            await output_list.remove_children()
            if outputs:
                await output_list.mount(plain("Saved files", classes="role"))
            for index, artifact in enumerate(outputs):
                await output_list.mount(Vertical(plain(f"{artifact['path']}\n{artifact.get('format', '').upper()} · {artifact.get('size_bytes', 0):,} bytes"), Button("Copy path", id=f"output-path-{index}", classes="output-copy", compact=True), classes="output-card"))
            if outputs:
                await self.load_files(path=self._file_root or None, query=self.query_one("#file-filter", Input).value)
        if at_end:
            container.call_after_refresh(container.scroll_end, animate=False)
        else:
            container.call_after_refresh(container.scroll_to, y=previous_y, animate=False)

    async def render_chat_list(self) -> None:
        conversations = records(self.snapshot.get("conversations"))
        key = repr([(c["id"], c.get("title"), c["id"] == self.conversation_id) for c in conversations])
        if key == self._chat_list_key:
            return
        self._chat_list_key = key
        panel = self.query_one("#chat-list", VerticalScroll)
        await panel.remove_children()
        for conversation in reversed(conversations):
            await panel.mount(Button(safe(conversation.get("title", "New chat"))[:60], id=f"open-chat-{conversation['id']}", variant="primary" if conversation["id"] == self.conversation_id else "default"))

    def load_draft(self, conversation: dict) -> None:
        self._loading_draft = True
        self.query_one("#composer", TextArea).load_text(conversation.get("draft", ""))
        self._last_sent_draft = conversation.get("draft", "")
        self._draft_dirty = False
        self._loading_draft = False
        self.attachment_ids = list(conversation.get("attachment_ids", []))
        self.source_ids = []

    async def open_conversation(self, id_: str, *, flush: bool = True) -> None:
        if flush:
            await self.save_draft()
        self.conversation_id = id_
        self.load_draft(self.conversation)
        self._render_key = ""

    @on(TextArea.Changed, "#composer")
    def draft_changed(self) -> None:
        if self._loading_draft or self._detaching:
            return
        self._draft_dirty = True
        if self._draft_timer:
            self._draft_timer.stop()
        self._draft_timer = self.set_timer(0.3, self.save_draft)

    async def save_draft(self) -> bool:
        # Serialize a pending autosave and the final save before leaving.
        async with self._draft_lock:
            if not self.conversation_id or not self.screen_stack or not self.screen_stack[0].query("#composer"):
                return True
            draft = self.screen_stack[0].query_one("#composer", TextArea).text
            if draft == self._last_sent_draft:
                self._draft_dirty = False
                return True
            try:
                await self.api("PATCH", f"/conversations/{self.conversation_id}", json={"draft": draft})
                self._last_sent_draft = draft
                self._draft_dirty = False
                return True
            except Exception as exc:
                self.report_error(f"Draft is still on screen but could not be saved: {safe(exc)}")
                return False

    @on(ChatComposer.Submitted)
    async def composer_submitted(self) -> None:
        await self.action_send(cancel_active=False)

    async def action_send(self, operation: str = "chat", text: str | None = None, retry_of: str | None = None, *, cancel_active: bool = True) -> None:
        if self.screen is not self.screen_stack[0]:
            return
        if self._settings_operation and not self.snapshot.get("active_turn_id"):
            return
        try:
            active = self.snapshot.get("active_turn_id")
            if active:
                if not cancel_active:
                    return
                await self.api("POST", f"/turns/{active}/cancel", json={})
            else:
                composer = self.query_one("#composer", TextArea)
                message = text if text is not None else composer.text
                if not message.strip():
                    composer.focus()
                    return
                await self.save_draft()
                await self.api("POST", "/turns", json={"conversation_id": self.conversation_id, "text": message, "attachment_ids": self.attachment_ids, "source_ids": self.source_ids, "operation": operation, "retry_of": retry_of})
                self._loading_draft = True
                composer.load_text("")
                self._loading_draft = False
                self._draft_dirty = True
                await self.save_draft()
                self.last_error = ""
                self.source_ids = []
            await self.refresh_state()
        except Exception as exc:
            self.report_error(exc)
            if getattr(exc, "code", "") == "context_overflow":
                self.push_screen(ConfirmDialog("Context is full", "Start a new chat for fresh context. Your full transcript and current draft stay saved. You can also choose fewer files, set an explicit history limit in Advanced, or load a larger context. Nothing was silently dropped.", "New chat"), lambda yes: self.run_worker(self.action_new_chat()) if yes else None)

    async def action_new_chat(self) -> None:
        if len(self.screen_stack) > 1:
            return
        try:
            await self.save_draft()
            conversation = await self.api("POST", "/conversations", json={})
            self.accept_snapshot(await self.client.state())
            await self.open_conversation(conversation["id"], flush=False)
            await self.refresh_state()
        except Exception as exc:
            self.report_error(exc)

    def last_reply(self) -> str:
        return next((turn["answer"] for turn in reversed(self.conversation.get("turns", [])) if turn.get("answer")), "")

    def copy_reply(self) -> None:
        if reply := self.last_reply():
            self.copy_to_clipboard(safe(reply))
            self.query_one("#paste-copied").display = True
            self.query_one("#composer", TextArea).focus()
            self.notify("Copied in this app; terminal clipboard requested. Ctrl+V pastes the copied text.", timeout=4)

    def paste_copied(self) -> None:
        composer = self.query_one("#composer", TextArea)
        composer.action_paste()
        composer.focus()

    def action_panel(self) -> None:
        if len(self.screen_stack) > 1:
            return
        panel = self.query_one("#drawer")
        panel.display = not panel.display
        self._panel_choice = panel.display

    def on_resize(self, event: Resize) -> None:
        if self.query("#drawer"):
            self.query_one("#drawer").display = event.size.width >= 108 if self._panel_choice is None else self._panel_choice

    def action_settings(self) -> None:
        if self.manifest and len(self.screen_stack) == 1 and not self._settings_operation:
            self.push_screen(SettingsDialog(self.manifest), self.settings_result)

    def settings_result(self, result: dict | None) -> None:
        if not result:
            return
        if result.get("new_allocation"):
            self.push_screen(ConfirmDialog("Return to allocation selection", "Detach to choose another model or resources. The current GPU session keeps running until you stop it.", "Detach"), lambda yes: self.run_worker(self.detach(new_session=True)) if yes else None)
            return
        settings = result["settings"]
        current = self.manifest["settings"]
        runtime = ["context", "acceleration"] + [v[0] for v in RUNTIME + ACCELERATION] + ["cache_type_k", "cache_type_v", "flash_attention"]
        reload_needed = any(current.get(k) != settings.get(k) for k in runtime)
        if reload_needed:
            disclosure = "\n\n" + self.web_enable_message(settings) if settings.get("web_enabled") and not current.get("web_enabled") else ""
            self.push_screen(ConfirmDialog("Reload the model?", "Context/runtime changes reload the model in this allocation and may need more memory. A failed reload preserves the previous settings. Active responses must finish first." + disclosure, "Reload"), lambda yes: self.run_worker(self.apply_settings(settings, confirm_reload=True, save_model_default=result.get("save_model_default", False))) if yes else None)
        elif settings.get("web_enabled") and not current.get("web_enabled"):
            self.push_screen(ConfirmDialog("Enable Web?", self.web_enable_message(settings), "Enable"), lambda yes: self.run_worker(self.apply_settings(settings, save_model_default=result.get("save_model_default", False))) if yes else None)
        else:
            self.run_worker(self.apply_settings(settings, save_model_default=result.get("save_model_default", False)))

    @staticmethod
    def web_enable_message(settings: dict) -> str:
        permission = "Each query or URL will ask for your approval." if settings.get("web_approval") == "per_request" else "Searches and page requests may run without separate prompts until you turn Web Off."
        return "Queries and requested URLs go to external services. " + permission + " Inference and file parsing stay local, but derived queries can contain private terms. You can enable per-request review in Advanced."

    async def apply_settings(self, settings: dict, *, confirm_reload: bool = False, save_model_default: bool = False) -> None:
        if self._settings_operation:
            return
        self._settings_operation = "Reloading model…" if confirm_reload else "Applying settings…"
        self.update_header()
        try:
            manifest = await self.api("PUT", "/settings", json={"settings": settings, "confirm_reload": confirm_reload, "save_model_default": save_model_default})
            # PUT returns only after validation/reload succeeds. Commit that
            # authoritative result even when the periodic GET is still pending.
            self.accept_snapshot(dict(self.snapshot, manifest=manifest))
            self.last_error = ""
            self._settings_operation = ""
            self.update_header()
            await self.refresh_state()
            self.notify("Settings saved" + (" as model default" if save_model_default else ""))
        except Exception as exc:
            self.report_error(exc)
        finally:
            self._settings_operation = ""
            self.update_header()

    def action_attach(self) -> None:
        if len(self.screen_stack) == 1:
            self.push_screen(AttachDialog(), self.attach_result)

    def attach_result(self, result: dict | None) -> None:
        if result:
            self.run_worker(self.attach_file(result))

    async def attach_file(self, result: dict) -> None:
        try:
            self.query_one("#activity", Static).update("Reading local file…")
            attachment = await self.api("POST", "/attachments", json=result)
            if attachment.get("status") not in {"error", "unsupported"}:
                self.attachment_ids = list(dict.fromkeys(self.attachment_ids + [attachment["id"]]))
            self.accept_snapshot(await self.client.state())
            self.update_tray()
            self.push_screen(AttachmentsDialog([attachment], self.attachment_ids), lambda selected: self.attachment_preview_result(selected, attachment["id"]))
        except Exception as exc:
            self.report_error(exc)

    def attachment_preview_result(self, result: list[str] | None, attachment_id: str) -> None:
        if result is not None:
            # A single-file preview may add/remove that file without dropping others.
            self.attachment_ids = list(dict.fromkeys([i for i in self.attachment_ids if i != attachment_id] + result))
            self.update_tray()

    def attachments_result(self, result: list[str] | None) -> None:
        if result is not None:
            self.attachment_ids = result
            self.update_tray()

    def update_tray(self) -> None:
        self.query_one("#tray").display = bool(self.attachment_ids or self.source_ids)
        self.query_one("#tray-files").display = bool(self.attachment_ids)
        self.query_one("#tray-sources").display = bool(self.source_ids)
        self.query_one("#tray-files", Button).label = f"{len(self.attachment_ids)} files · review"
        self.query_one("#tray-sources", Button).label = f"{len(self.source_ids)} sources · review"

    async def load_files(self, path: str | None = None, query: str = "") -> None:
        try:
            result = await self.api("GET", "/files", params={k: v for k, v in {"path": path, "query": query}.items() if v})
            entries = result.get("entries", result.get("files", [])) if isinstance(result, dict) else result
            tree = self.query_one("#file-tree", Tree)
            tree.clear()
            self._tree_loaded.clear()
            self._file_root = path or (result.get("path", result.get("workspace", "")) if isinstance(result, dict) else self._file_root)
            tree.root.set_label(safe(Path(self._file_root).name if self._file_root else "Workspace"))
            tree.root.data = {"path": self._file_root, "is_dir": True}
            self.add_file_nodes(tree.root, entries)
            tree.root.expand()
        except Exception as exc:
            self.report_error(exc)

    def add_file_nodes(self, parent, entries: list[dict]) -> None:
        for entry in entries:
            if not entry.get("path") or entry.get("kind") in {"notice", "symlink", "special"}:
                parent.add(safe(entry.get("name", "Unavailable entry")), data=None, allow_expand=False)
                continue
            is_dir = entry.get("is_dir", entry.get("type") == "directory" or entry.get("kind") == "directory")
            entry = dict(entry, is_dir=is_dir)
            parent.add(safe(entry.get("name", Path(entry["path"]).name)), data=entry, allow_expand=is_dir)

    @on(Tree.NodeExpanded, "#file-tree")
    async def expand_directory(self, event: Tree.NodeExpanded) -> None:
        node = event.node
        if node is self.query_one("#file-tree", Tree).root or not node.data:
            return
        path = node.data.get("path")
        if not path or path in self._tree_loaded:
            return
        self._tree_loaded.add(path)
        try:
            result = await self.api("GET", "/files", params={"path": path})
            entries = result.get("entries", result.get("files", [])) if isinstance(result, dict) else result
            self.add_file_nodes(node, entries)
        except Exception as exc:
            self._tree_loaded.discard(path)
            self.report_error(exc)

    @on(Tree.NodeSelected, "#file-tree")
    def choose_file(self, event: Tree.NodeSelected) -> None:
        if event.node.data and not event.node.data.get("is_dir"):
            self.push_screen(AttachDialog(event.node.data["path"]), self.attach_result)

    @on(Input.Submitted, "#file-filter")
    async def filter_files(self, event: Input.Submitted) -> None:
        await self.load_files(query=event.value)

    def workspace_result(self, path: str | None) -> None:
        if path:
            self.run_worker(self.change_workspace(path))

    async def change_workspace(self, path: str) -> None:
        try:
            await self.api("PUT", "/workspace", json={"path": path})
            self._file_root = path
            await self.load_files(path)
        except Exception as exc:
            self.report_error(exc)

    async def check_approvals(self) -> None:
        if len(self.screen_stack) > 1:
            return
        for approval in records(self.snapshot.get("approvals")):
            if approval.get("conversation_id") != self.conversation_id:
                continue
            if approval["status"] == "pending" and approval["id"] not in self._seen_approvals:
                self._seen_approvals.add(approval["id"])
                self.push_screen(ApprovalDialog(approval), lambda result, a=approval: self.run_worker(self.approval_result(a, result)))
                return
            if approval["status"] in {"completed", "failed"} and approval["id"] not in self._seen_results:
                self._seen_results.add(approval["id"])
                if approval.get("turn_id"):
                    continue
                if approval["status"] == "failed":
                    self.report_error(approval.get("error") or "External request failed; offline chat remains available.")
                    continue
                result = approval.get("result") or {}
                sources = result.get("sources", []) if isinstance(result, dict) else result
                if isinstance(sources, list) and sources:
                    self.push_screen(SourceSelectionDialog(sources), self.sources_result)
                    return
                self.notify("No sources returned. Try a revised query; offline chat remains available.")

    async def approval_result(self, approval: dict, result: dict | None) -> None:
        try:
            approve = bool(result and result.get("approve"))
            edited = approve and result["arguments"] != approval["arguments"]
            if edited:
                approval = await self.api("POST", f"/approvals/{approval['id']}", json={"edited_arguments": result["arguments"], "action_hash": approval["action_hash"]})
            await self.api("POST", f"/approvals/{approval['id']}", json={"approve": approve, "action_hash": approval["action_hash"]})
            await self.refresh_state()
        except Exception as exc:
            self.report_error(exc)

    def search_result(self, query: str | None) -> None:
        if query and query.strip():
            self.run_worker(self.request_tool("web_search", {"query": query.strip()}))

    async def request_tool(self, name: str, arguments: dict) -> None:
        try:
            await self.api("POST", "/tools/request", json={"conversation_id": self.conversation_id, "name": name, "arguments": arguments})
            await self.refresh_state()
        except Exception as exc:
            self.report_error(exc)

    def sources_result(self, result: dict | None, replace: bool = False) -> None:
        if not result:
            return
        if result.get("fetch"):
            self.run_worker(self.request_tool("fetch_public_page", {"url": result["fetch"]}))
        else:
            self.source_ids = list(dict.fromkeys(([] if replace else self.source_ids) + result.get("source_ids", [])))
            self.update_tray()

    def show_provider(self) -> None:
        self.run_worker(self.provider_status())

    async def provider_status(self) -> None:
        try:
            status = await self.api("GET", "/web/status")
            self.push_screen(ValueDialog("Web provider", safe(status) + "\nTo configure Brave, enter your own API key below. Leave blank to keep it. A provider plan may incur charges; no account is created.", password=True), self.provider_key_result)
        except Exception as exc:
            self.report_error(exc)

    def provider_key_result(self, key: str | None) -> None:
        if key and key.strip():
            self.run_worker(self.save_key(key.strip()))

    async def save_key(self, key: str) -> None:
        try:
            await self.api("POST", "/web/key", json={"key": key})
            self.notify("Provider key saved privately")
        except Exception as exc:
            self.report_error(exc)

    def export_result(self, result: dict | None) -> None:
        if result:
            self.run_worker(self.export_chat(result))

    async def export_chat(self, result: dict) -> None:
        try:
            result = await self.api("POST", "/export", json=dict(result, conversation_id=self.conversation_id))
            self.notify(safe(f"Saved {result['path']}"), timeout=8)
        except Exception as exc:
            self.report_error(exc)

    def open_reply(self, turn_id: str) -> None:
        turn = next((turn for turn in self.conversation.get("turns", []) if turn["id"] == turn_id), None)
        if turn and turn.get("answer"):
            self.push_screen(ReplyDialog(turn), self.reply_result)

    def reply_result(self, turn_id: str | None) -> None:
        if turn_id:
            self.push_screen(SaveReplyDialog(turn_id, self.snapshot.get("workspace", self._file_root)), self.save_reply_result)

    def save_reply_result(self, result: dict | None) -> None:
        if result:
            self.run_worker(self.save_reply(result))

    async def save_reply(self, result: dict) -> None:
        try:
            saved = await self.api("POST", "/export/reply", json=dict(result, conversation_id=self.conversation_id))
            self.accept_snapshot(await self.client.state())
            await self.refresh_state()
            self.notify(safe(f"Saved reply: {saved['path']}"), timeout=8)
        except Exception as exc:
            message = safe(getattr(exc, "message", str(exc)))
            if any(value in message for value in ["Unknown API", "Unknown route", "Unknown application action"]):
                message = "This running supervisor predates Save reply. Update the app and start a new GPU session; saved chats can be restored. No file was saved."
            self.report_error(message)

    async def delete_chat(self, yes: bool) -> None:
        if yes:
            try:
                await self.api("DELETE", f"/conversations/{self.conversation_id}")
                self.conversation_id = None
                await self.refresh_state()
            except Exception as exc:
                self.report_error(exc)

    async def retry_last(self, yes: bool) -> None:
        if not yes:
            return
        turns = self.conversation.get("turns", [])
        if turns:
            previous = turns[-1]
            await self.action_send(previous.get("request", {}).get("operation", "chat"), previous["request"]["text"], previous["id"])

    async def action_quit(self) -> None:
        # Textual's inherited Ctrl+Q otherwise bypasses the final draft save.
        await self.detach()

    async def action_detach(self) -> None:
        await self.detach()

    def pause_session_timers(self) -> None:
        if self._refresh_timer:
            self._refresh_timer.pause()
        if self._draft_timer:
            self._draft_timer.stop()
            self._draft_timer = None

    async def on_unmount(self) -> None:
        self._detaching = True
        self.pause_session_timers()
        self.workers.cancel_all()
        await self.client.close()

    async def detach(self, new_session: bool = False, *, discard_draft: bool = False) -> None:
        if self._detaching or self._detach_prompt:
            return
        # Mark departure before awaiting I/O, so repeats cannot start another exit.
        self._detaching = True
        self.pause_session_timers()
        saved = discard_draft
        if not discard_draft:
            try:
                saved = await asyncio.wait_for(self.save_draft(), self.DETACH_SAVE_TIMEOUT)
            except TimeoutError:
                self.report_error("Draft save timed out. Your draft is still on screen.")
        if not saved:
            self._detaching = False
            if self._refresh_timer:
                self._refresh_timer.resume()
            self._detach_prompt = True

            def confirm_discard(yes: bool) -> None:
                self._detach_prompt = False
                if yes:
                    self.run_worker(self.detach(new_session, discard_draft=True))

            self.push_screen(ConfirmDialog(
                "Draft was not saved",
                "Stay here to copy your draft or retry after reconnecting. Detaching now loses the unsaved draft. The GPU session and response keep running.",
                "Detach without saving",
            ), confirm_discard)
            return
        # Close HTTP during Textual teardown, not while this screen is still running.
        self.exit({"action": "new_session" if new_session else "detach"})

    async def stop_session(self, yes: bool) -> None:
        if yes:
            try:
                await self.save_draft()
                await self.api("POST", "/stop", json={})
                self._detaching = True
                self.pause_session_timers()
                self.exit({"action": "stop"})
            except Exception as exc:
                self.report_error(exc)

    def action_help(self) -> None:
        self.push_screen(TextDialog("Quick help", "Enter sends from the message box. Shift+Enter adds a line; use Ctrl+J if your terminal cannot distinguish Shift+Enter. Ctrl+S sends or explicitly stops a reply; Enter never stops it.\n\nPaste outside text using your terminal's paste command (often Ctrl+Shift+V or Shift+Insert), or the Open OnDemand clipboard controls. Bracketed multiline paste stays in the draft until you send it.\nSelect text and Ctrl+C copies it; Ctrl+X cuts selected composer text. Copy reply copies the latest answer. Copy requests the terminal clipboard and also keeps an in-app copy; browser/terminal permissions may block the external clipboard. Ctrl+V or Paste copied pastes only text copied inside this app, not your laptop clipboard.\n\nCtrl+O attaches a local file. Ctrl+B opens Files / Chats.\nCtrl+N or New chat starts an empty conversation; the old chat stays saved. Ctrl+P opens Settings.\nCtrl+D or Ctrl+Q detaches; the GPU session and any response continue.\n\nChats save automatically. Cancel response only cancels that response. Delete chat only deletes that saved conversation. Stop GPU session (Chats panel) explicitly ends the allocation.\n\nWeb starts Off. Enabling Web permits searches and page requests until you turn it Off. Optional per-request review is in Advanced. Sources disclose snippets versus fetched content. Thinking generation and showing reasoning are separate controls.\n\nContext shows loaded capacity; changing it requires a confirmed model reload. Maximum response 0 means Auto: use remaining context. Temperature and other sampling controls are in Settings → Advanced."))

    @on(Button.Pressed)
    async def buttons(self, event: Button.Pressed) -> None:
        id_ = event.button.id or ""
        if self._settings_operation and id_ in {"thinking", "context", "web", "settings"}:
            return
        if id_.startswith("open-chat-"):
            await self.open_conversation(id_.removeprefix("open-chat-"))
            await self.refresh_state()
        elif id_.startswith("open-reply-"):
            self.open_reply(id_.removeprefix("open-reply-"))
        elif id_.startswith("output-path-"):
            index = int(id_.removeprefix("output-path-"))
            self.copy_to_clipboard(self.conversation["outputs"][index]["path"])
            self.notify("Copied path in this app; terminal clipboard requested.")
        elif id_ == "send":
            await self.action_send()
        elif id_ == "copy-reply":
            self.copy_reply()
        elif id_ == "paste-copied":
            self.paste_copied()
        elif id_ == "panel":
            self.action_panel()
        elif id_ in {"new-chat", "new-chat-main"}:
            await self.action_new_chat()
        elif id_ == "settings":
            self.action_settings()
        elif id_ == "context":
            self.push_screen(SettingsDialog(self.manifest, focus_context=True), self.settings_result)
        elif id_ == "thinking":
            self.push_screen(ThinkingDialog(self.manifest["settings"], self.manifest.get("capabilities", {})), lambda result: self.run_worker(self.apply_settings(dict(self.manifest["settings"], **result))) if result else None)
        elif id_ == "web":
            settings = dict(self.manifest["settings"])
            settings["web_enabled"] = not settings["web_enabled"]
            self.settings_result({"settings": settings})
        elif id_ == "search":
            if not self.manifest["settings"]["web_enabled"]:
                self.notify("Turn Web on first to allow external searches and page requests.", timeout=5)
            else:
                self.push_screen(ValueDialog("Search web", f"Search with {self.manifest['settings']['web_provider']}. This query goes to the provider; no conversation or attachment is sent automatically.", action_label="Search"), self.search_result)
        elif id_ == "attach":
            self.action_attach()
        elif id_ in {"attachments", "tray-files"}:
            self.push_screen(AttachmentsDialog(records(self.snapshot.get("attachments")), self.attachment_ids), self.attachments_result)
        elif id_ == "tray-sources":
            sources = self.conversation.get("sources", {})
            self.push_screen(SourceSelectionDialog([s for s in records(sources) if s["id"] in self.source_ids]), lambda result: self.sources_result(result, replace=True))
        elif id_ == "workspace":
            self.push_screen(ValueDialog("Choose workspace", "Explicitly choose a directory for browsing, attachments and exports. Symlink and path rules are checked by the supervisor.", self._file_root), self.workspace_result)
        elif id_ == "file-actions":
            self.push_screen(FileActionsDialog([a for a in records(self.snapshot.get("attachments")) if a["id"] in self.attachment_ids]), self.file_action_result)
        elif id_ == "export":
            self.push_screen(ExportDialog("chat-export.md"), self.export_result)
        elif id_ == "retry":
            self.push_screen(ConfirmDialog("Retry last message?", "Retry appends a new answer with the current settings. The supervisor excludes the previous attempt from the retry's model history; both attempts remain saved.", "Retry"), lambda yes: self.run_worker(self.retry_last(yes)))
        elif id_ == "delete-chat":
            self.push_screen(ConfirmDialog("Delete saved chat?", "Deletes this conversation. It does not stop the GPU allocation. A chat with an active response cannot be deleted.", "Delete"), lambda yes: self.run_worker(self.delete_chat(yes)))
        elif id_ == "detach":
            await self.action_detach()
        elif id_ == "stop-session":
            self.push_screen(ConfirmDialog("Stop GPU session?", "Ends the server and this GPU allocation. Saved conversations stay on disk. To leave it running, choose Detach instead.", "Stop session"), lambda yes: self.run_worker(self.stop_session(yes)))

    def file_action_result(self, result: dict | None) -> None:
        if result:
            self.run_worker(self.file_action(result))

    async def file_action(self, result: dict) -> None:
        if result["operation"] == "table":
            try:
                answer = await self.api("POST", "/files/table", json={"conversation_id": self.conversation_id, "attachment_id": result["attachment_id"], "operation": result["table_operation"], "arguments": result["arguments"]})
                await self.refresh_state()
                self.push_screen(TextDialog("Deterministic table result", "Saved with this chat. Use Chats → Save result to export.\n\n" + json.dumps(answer, indent=2, ensure_ascii=False)))
            except Exception as exc:
                self.report_error(exc)
        else:
            default = {"summarize": "Summarize the selected file excerpts, explain coverage limits and cite source IDs.", "compare": "Compare the selected files and cite source IDs for each difference.", "extract": "Extract the requested information from the selected files with source IDs.", "draft_query": "Draft a short web search query for my question. Do not search yet."}[result["operation"]]
            await self.action_send(result["operation"], text=self.query_one("#composer", TextArea).text or default)


class FileActionsDialog(Dialog):
    def __init__(self, attachments: list[dict]):
        super().__init__()
        self.attachments = attachments

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain("File actions", classes="dialog-title")
            yield plain("Uses your composer instructions and the selected file snapshots. Numeric table operations are deterministic, not model guesses.", classes="help")
            yield Select([("Summarize", "summarize"), ("Compare", "compare"), ("Extract", "extract"), ("Draft search query (local model)", "draft_query"), ("Table operation", "table")], value="summarize", allow_blank=False, id="file-operation")
            with VerticalScroll(classes="dialog-body", id="table-details"):
                yield Label("Table file")
                yield Select([(safe(a["name"]), a["id"]) for a in self.attachments], value=self.attachments[0]["id"] if self.attachments else Select.NULL, id="table-attachment")
                yield Select([("Count rows", "count"), ("Numeric summary / missing values", "summary"), ("Filter rows", "filter"), ("Group rows", "group")], value="count", allow_blank=False, id="table-operation")
                yield Label("Column for numeric summary (optional for all numeric columns)")
                yield Input(id="table-column", placeholder="value")
                yield Label("Group by column (Group rows)")
                yield Input(id="table-group-by", placeholder="treatment")
                yield Label("Filter column / comparison / value (Filter rows)")
                yield Input(id="table-filter-column", placeholder="value")
                yield Select([("Equals", "eq"), ("Not equal", "ne"), ("Greater than", "gt"), ("At least", "gte"), ("Less than", "lt"), ("At most", "lte"), ("Contains", "contains"), ("Is missing", "missing")], value="eq", allow_blank=False, id="table-filter-op")
                yield Input(id="table-filter-value", placeholder="18")
                yield Label("Sheet / table name (optional)")
                yield Input(id="table-sheet")
            yield plain("", id="file-action-error", classes="error")
            with Horizontal(classes="actions"):
                yield Button("Run", variant="primary", id="file-action-run")
                yield Button("Cancel", id="file-action-cancel")

    def on_mount(self) -> None:
        self.query_one("#table-details").display = False

    @on(Select.Changed, "#file-operation")
    def operation_changed(self, event: Select.Changed) -> None:
        self.query_one("#table-details").display = event.value == "table"

    @on(Button.Pressed)
    def answer(self, event: Button.Pressed) -> None:
        if event.button.id == "file-action-cancel":
            self.dismiss(None)
        elif event.button.id == "file-action-run":
            try:
                operation = self.query_one("#file-operation", Select).value
                table_operation = self.query_one("#table-operation", Select).value
                attachment_id = self.query_one("#table-attachment", Select).value
                arguments = {}
                if operation == "table":
                    if attachment_id is Select.NULL:
                        raise ValueError("Choose an attached CSV, TSV or XLSX file first.")
                    fields = [("table-sheet", "sheet")]
                    if table_operation in {"summary", "group"}:
                        fields.append(("table-column", "column"))
                    if table_operation == "group":
                        fields.append(("table-group-by", "group_by"))
                    for field, key in fields:
                        value = self.query_one(f"#{field}", Input).value.strip()
                        if value:
                            arguments[key] = value
                    if table_operation == "group" and not arguments.get("group_by"):
                        raise ValueError("Choose the column to group by.")
                    if table_operation == "filter":
                        column = self.query_one("#table-filter-column", Input).value.strip()
                        if not column:
                            raise ValueError("Choose the column to filter.")
                        raw = self.query_one("#table-filter-value", Input).value
                        try:
                            value = json.loads(raw)
                        except ValueError:
                            value = raw
                        arguments["where"] = {"column": column, "op": self.query_one("#table-filter-op", Select).value, "value": value}
                self.dismiss({"operation": operation, "attachment_id": str(attachment_id), "table_operation": table_operation, "arguments": arguments})
            except ValueError as exc:
                self.query_one("#file-action-error", Static).update(safe(exc))


class ResourceDialog(Dialog):
    DEFAULT_CSS = """
    ResourceDialog SelectCurrent { height: 3; }
    ResourceDialog SelectCurrent Static#label { height: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    """

    def __init__(self, resources: dict, inventory: dict | None = None, profile: dict | None = None):
        super().__init__()
        self.resources = resources
        self.inventory = inventory or {}
        self.profile = data(profile) or {}
        self.presets = self.profile.get("resource_presets", {})

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain("Resources for a new allocation", classes="dialog-title")
            with VerticalScroll(classes="dialog-body"):
                yield plain("No job starts until you choose Start session. These are requests, not a guarantee that the model fits.", classes="help")
                if self.presets:
                    yield Label("Start from a saved preset")
                    yield Select([("Custom request", "")] + [(safe(name), name) for name in self.presets], value="", allow_blank=False, id="resource-preset")
                if self.inventory.get("available"):
                    yield plain("Discovered (cached): " + "; ".join(f"{row.get('name', '?')}: {row.get('gres') or 'GPU details unavailable'}" for row in self.inventory.get("partitions", [])), classes="help")
                else:
                    yield plain("Using profile defaults; live inventory unavailable. You can edit the request below.", classes="help")
                for name, title, kind in RESOURCE_FIELDS:
                    yield Label("Queue" if name == "partition" and self.profile.get("scheduler") == "pbs" else title)
                    yield Input(str(self.resources.get(name, "")), id=f"resource-{name}", type="integer" if kind is int else "text")
            yield plain("", id="resource-error", classes="error")
            with Horizontal(classes="actions"):
                yield Button("Reset", id="resource-reset")
                yield Button("Cancel", id="resource-cancel")
                yield Button("Apply", variant="primary", id="resource-apply")

    @on(Select.Changed, "#resource-preset")
    def choose_preset(self, event: Select.Changed) -> None:
        if event.value in self.presets:
            for name, _, _ in RESOURCE_FIELDS:
                self.query_one(f"#resource-{name}", Input).value = str(data(self.presets[event.value]).get(name, ""))

    @on(Button.Pressed)
    def answer(self, event: Button.Pressed) -> None:
        if event.button.id == "resource-cancel":
            self.dismiss(None)
        elif event.button.id == "resource-reset":
            defaults = self.profile.get("resources") or ResourceRequest().model_dump()
            for name, _, _ in RESOURCE_FIELDS:
                self.query_one(f"#resource-{name}", Input).value = str(defaults[name])
        elif event.button.id == "resource-apply":
            try:
                self.dismiss(ResourceRequest(**{name: kind(self.query_one(f"#resource-{name}", Input).value) for name, _, kind in RESOURCE_FIELDS}).model_dump())
            except (ValueError, TypeError) as exc:
                self.query_one("#resource-error", Static).update(safe(exc))


class SiteSetupDialog(Dialog):
    DEFAULT_CSS = """
    SiteSetupDialog > Vertical { width: 96; height: 94%; }
    SiteSetupDialog TabbedContent { height: 1fr; min-height: 3; }
    SiteSetupDialog #site-error { max-height: 3; overflow-y: auto; }
    SiteSetupDialog > Vertical > .actions { height: 3; min-height: 3; }
    SiteSetupDialog SelectCurrent { height: 3; }
    SiteSetupDialog SelectCurrent Static#label { height: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    """

    def __init__(self, service, profile: dict):
        super().__init__()
        self.service = service
        self.pending = SiteProfile(**data(profile)).model_dump()
        self.profiles = [data(item) for item in service.list()]
        if not any(item["name"] == self.pending["name"] for item in self.profiles):
            self.profiles.append(self.pending)
        self.cache_override = service.config().get("model_cache")
        self.preset_resources = dict(self.pending["resources"])

    @staticmethod
    def summary(resources: dict) -> str:
        return f"{resources['partition']} · {resources['gpu_type'] or 'GPU'} × {resources['gpu_count']} · {resources['cpus']} CPUs · {resources['memory_gb']} GB · {resources['walltime']}"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain("Site setup", classes="dialog-title")
            with TabbedContent():
                with TabPane("Profile", id="site-profile-tab"):
                    with VerticalScroll():
                        yield Label("Start from a saved site profile")
                        yield Select([(safe(f"{item['name']} · {item['scheduler'].upper()}"), item["name"]) for item in self.profiles], value=self.pending["name"], allow_blank=False, id="site-profile")
                        yield Label("Profile name (change the name to make a copy)")
                        yield Input(self.pending["name"], id="site-name")
                        yield Label("Description")
                        yield Input(self.pending.get("description", ""), id="site-description")
                        yield Label("Scheduler")
                        yield Select([("Slurm", "slurm"), ("PBS", "pbs")], value=self.pending["scheduler"], allow_blank=False, id="site-scheduler")
                        yield Label("Model cache directory")
                        yield Input(self.pending.get("model_cache", ""), id="site-cache")
                        yield plain("Existing model files stay where they are. Relative paths use the installation folder.", classes="help")
                        if os.environ.get("HPC_LLM_MODEL_CACHE"):
                            yield plain(f"HPC_LLM_MODEL_CACHE currently overrides this setting: {os.environ['HPC_LLM_MODEL_CACHE']}. Your saved choice takes effect after that environment variable is unset.", id="site-cache-override", classes="help")
                        yield Label("Default resources")
                        yield Button(self.summary(self.pending["resources"]), id="site-defaults")
                        with Collapsible(title="Runtime / PBS details", collapsed=True):
                            yield Label("Runtime executable (blank uses the bundled runtime)")
                            yield Input(self.pending.get("runtime", ""), id="site-runtime")
                            yield Label("PBS GPU count resource key")
                            yield Input(self.pending["pbs_gpu_resource"], id="site-pbs-gpu")
                            yield Label("PBS GPU type resource key (optional, site-specific)")
                            yield Input(self.pending["pbs_gpu_type_resource"], id="site-pbs-type")
                            yield Label("PBS attach command")
                            yield Input(self.pending["pbs_attach_command"], id="site-pbs-attach")
                with TabPane("Resource presets", id="site-presets-tab"):
                    with VerticalScroll():
                        yield plain("Save named queue, GPU and duration choices for the Resources menu.", classes="help")
                        yield Select([(safe(name), name) for name in self.pending["resource_presets"]], id="site-preset", prompt="Choose a preset to edit")
                        yield Label("Preset name")
                        yield Input(id="site-preset-name", placeholder="Short GPU session")
                        yield plain(self.summary(self.preset_resources), id="site-preset-summary")
                        yield Button("Edit preset resources…", id="site-preset-resources")
                        with Horizontal(classes="actions"):
                            yield Button("Save preset", id="site-preset-save")
                            yield Button("Delete preset", id="site-preset-delete")
                        yield plain("Changes stay pending until Save & use below.", classes="help")
            yield plain("", id="site-error", classes="error")
            with Horizontal(classes="actions"):
                yield Button("Save & use", id="site-save", variant="primary")
                yield Button("Cancel", id="site-cancel")

    @on(Select.Changed, "#site-profile")
    def profile_selected(self, event: Select.Changed) -> None:
        profile = next((item for item in self.profiles if item["name"] == event.value), None)
        if not profile or not self.query("#site-name"):
            return
        self.pending = SiteProfile(**profile).model_dump()
        for selector, key in [("site-name", "name"), ("site-description", "description"), ("site-runtime", "runtime"), ("site-pbs-gpu", "pbs_gpu_resource"), ("site-pbs-type", "pbs_gpu_type_resource"), ("site-pbs-attach", "pbs_attach_command")]:
            self.query_one("#" + selector, Input).value = self.pending.get(key, "")
        self.query_one("#site-scheduler", Select).value = self.pending["scheduler"]
        self.query_one("#site-cache", Input).value = self.cache_override or self.pending.get("model_cache") or str(self.service.app_root / "cache" / "models")
        self.query_one("#site-defaults", Button).label = self.summary(self.pending["resources"])
        self.preset_resources = dict(self.pending["resources"])
        self.refresh_presets()

    @on(Select.Changed, "#site-preset")
    def preset_selected(self, event: Select.Changed) -> None:
        if event.value in self.pending["resource_presets"]:
            self.preset_resources = dict(self.pending["resource_presets"][event.value])
            self.query_one("#site-preset-name", Input).value = str(event.value)
            self.query_one("#site-preset-summary", Static).update(self.summary(self.preset_resources))

    def refresh_presets(self, selected=None) -> None:
        selector = self.query_one("#site-preset", Select)
        selector.set_options([(safe(name), name) for name in self.pending["resource_presets"]])
        selector.value = selected if selected in self.pending["resource_presets"] else Select.NULL

    def default_resources_result(self, resources: dict | None) -> None:
        if resources:
            self.pending["resources"] = resources
            self.query_one("#site-defaults", Button).label = self.summary(resources)

    def preset_resources_result(self, resources: dict | None) -> None:
        if resources:
            self.preset_resources = resources
            self.query_one("#site-preset-summary", Static).update(self.summary(resources))

    @on(Button.Pressed)
    async def setup_action(self, event: Button.Pressed) -> None:
        event.stop()
        id_ = event.button.id
        if id_ == "site-cancel":
            self.dismiss(None)
        elif id_ in {"site-defaults", "site-preset-resources"}:
            profile = dict(self.pending, scheduler=self.query_one("#site-scheduler", Select).value)
            resources = self.pending["resources"] if id_ == "site-defaults" else self.preset_resources
            self.app.push_screen(ResourceDialog(resources, profile=profile), self.default_resources_result if id_ == "site-defaults" else self.preset_resources_result)
        elif id_ == "site-preset-save":
            name = self.query_one("#site-preset-name", Input).value.strip()
            if not name or len(name) > 100 or any(ord(char) < 32 for char in name):
                self.query_one("#site-error", Static).update("Choose a preset name of 1–100 printable characters.")
                return
            self.pending["resource_presets"][name] = dict(self.preset_resources)
            self.refresh_presets(name)
        elif id_ == "site-preset-delete":
            selected = self.query_one("#site-preset", Select).value
            self.pending["resource_presets"].pop(selected, None)
            self.refresh_presets()
        elif id_ == "site-save":
            try:
                values = dict(self.pending, scheduler=self.query_one("#site-scheduler", Select).value)
                for selector, key in [("site-name", "name"), ("site-description", "description"), ("site-runtime", "runtime"), ("site-pbs-gpu", "pbs_gpu_resource"), ("site-pbs-type", "pbs_gpu_type_resource"), ("site-pbs-attach", "pbs_attach_command")]:
                    values[key] = self.query_one("#" + selector, Input).value.strip()
                profile = SiteProfile(**values)
                await asyncio.to_thread(self.service.save, profile, select=True, model_cache=self.query_one("#site-cache", Input).value.strip())
                self.dismiss({"action": "reload"})
            except Exception as exc:
                self.query_one("#site-error", Static).update(safe(getattr(exc, "message", str(exc))))


class ImportDialog(Dialog):
    DEFAULT_CSS = """
    ImportDialog SelectCurrent Static#label { height: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    """

    def __init__(self, service):
        super().__init__()
        import threading
        self.service = service
        self.models = [data(model) for model in service.list()] if service else []
        self.remote_files: list[dict] = []
        self.listed_repo = ""
        self.busy = False
        self.cancel_event = threading.Event()
        self.close_requested = False
        self.last_progress = 0.0
        self.install_plan = None
        self.preview_selection = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield plain("Models · install or add companions", classes="dialog-title")
            with VerticalScroll(classes="dialog-body"):
                yield Label("Install target")
                yield Select([("New model", "new")] + [(safe("Add companions: " + model["name"]), model["id"]) for model in self.models], value="new", allow_blank=False, id="model-target")
                yield plain("", id="model-target-info", classes="help")
                with Collapsible(title="Register existing local files", collapsed=False, id="model-local"):
                    yield Label("Existing local GGUF path")
                    yield Input(id="model-path", placeholder="/path/to/model.gguf")
                    yield Label("Matching vision projector path (optional)")
                    yield Input(id="model-projector", placeholder="/path/to/mmproj.gguf")
                    yield Label("Matching MTP head path (optional)")
                    yield Input(id="model-mtp", placeholder="/path/to/mtp.gguf")
                    yield Button("Register local model", id="model-register")
                yield Label("Download from Hugging Face: paste a model URL or owner/repository")
                yield Input(id="model-repo", placeholder="https://huggingface.co/owner/model-GGUF")
                yield Input(id="model-quant", placeholder="Quantization (optional), e.g. Q4_K_M")
                yield Button("Preview download", id="model-list")
                yield Select([], id="model-file", prompt="Choose a variant if requested")
                with Collapsible(title="Vision / acceleration and revision", collapsed=True, id="model-options"):
                    yield Checkbox("Full checksum recheck (slower)", id="model-verify-full")
                    yield plain("Unchanged files reuse a previous successful check. Enable this to read every byte again.", classes="help")
                    yield Label("Revision (branch, tag or immutable commit)")
                    yield Input("main", id="model-revision")
                    yield Label("Vision projector: auto includes a unique companion")
                    yield Select([("Automatic", "auto"), ("Skip / keep existing", "none")], value="auto", allow_blank=False, id="model-remote-projector")
                    yield Label("MTP acceleration head (optional)")
                    yield Select([("Skip / keep existing", "none"), ("Automatic matching head", "auto")], value="none", allow_blank=False, id="model-remote-mtp")
                yield plain("Preview shows exact downloads before installation. Add companions keeps the registered model, settings and existing companions; it does not download the main weights. Start a new allocation to use added companions.", classes="help")
                yield plain("", id="model-plan")
                yield Button("Download and register", id="model-download", disabled=True)
                yield plain("", id="model-progress")
            with Horizontal(classes="actions"):
                yield Button("Cancel download", id="model-cancel", disabled=True)
                yield Button("Close", id="model-close")

    def on_mount(self) -> None:
        self.query_one("#model-file", Select).display = False

    @on(Select.Changed, "#model-target")
    def target_changed(self, event: Select.Changed) -> None:
        model = next((m for m in self.models if m["id"] == event.value), None)
        self.query_one("#model-local").display = model is None
        self.query_one("#model-options", Collapsible).collapsed = model is None
        self.query_one("#model-quant").display = model is None
        self.query_one("#model-file").display = False
        self.query_one("#model-file", Select).value = Select.NULL
        self.query_one("#model-repo", Input).value = (model.get("repo_id") or "") if model else ""
        self.query_one("#model-revision", Input).value = (model.get("revision") or "main") if model else "main"
        self.query_one("#model-remote-projector", Select).value = "none" if model else "auto"
        self.query_one("#model-remote-mtp", Select).value = "none"
        self.install_plan = None
        self.query_one("#model-download", Button).disabled = True
        self.query_one("#model-plan", Static).update("")
        self.query_one("#model-target-info", Static).update(safe(
            f"Existing weights: {model['path']}\nVision: {model.get('projector_path') or 'not registered'}; MTP: {model.get('mtp_path') or ('embedded' if model.get('mtp_layers') else 'not registered')}"
            if model else ""))

    @on(Input.Changed, "#model-quant")
    def quant_changed(self, event: Input.Changed) -> None:
        # A newly typed quantization takes precedence over an older dropdown choice.
        self.query_one("#model-file", Select).value = Select.NULL

    @on(Button.Pressed)
    def buttons(self, event: Button.Pressed) -> None:
        id_ = event.button.id
        if id_ == "model-close":
            self.action_cancel()
            return
        if id_ == "model-cancel":
            self.cancel_event.set()
            self.query_one("#model-progress", Static).update("Cancelling after the current network read (up to 30 seconds). Partial bytes remain resumable.")
            return
        if self.busy:
            return
        if not self.service:
            self.query_one("#model-progress", Static).update("Model library is not available in this invocation. Use the CLI model registration command.")
            return
        if id_ in {"model-register", "model-list", "model-download"}:
            self.busy = True
            self.cancel_event.clear()
            self.close_requested = False
            for button in ("model-register", "model-list", "model-download"):
                self.query_one("#" + button, Button).disabled = True
            self.query_one("#model-cancel", Button).disabled = id_ != "model-download"
            self.run_worker(self.perform(id_), group="model-import")

    def action_cancel(self) -> None:
        if self.busy:
            self.close_requested = True
            self.cancel_event.set()
            self.query_one("#model-progress", Static).update("Finishing the current bounded operation before closing; no GPU job has started.")
        else:
            self.dismiss(None)

    def on_unmount(self) -> None:
        self.cancel_event.set()

    def download_progress(self, done: int, total: int) -> None:
        # Called on the download thread; marshal a throttled update to Textual.
        now = time.monotonic()
        if now - self.last_progress < .15 and done != total:
            return
        self.last_progress = now
        try:
            self.app.call_from_thread(self.show_progress, done, total)
        except (RuntimeError, asyncio.CancelledError):
            self.cancel_event.set()

    def show_progress(self, done: int, total: int) -> None:
        if self.cancel_event.is_set():
            return
        expected = f" / {total / 2**20:.1f} MiB" if total else " (total unknown)"
        self.query_one("#model-progress", Static).update(f"Downloaded {done / 2**20:.1f} MiB{expected}. Cancel keeps partial bytes for resume.")

    def install_selection(self):
        return (self.query_one("#model-repo", Input).value.strip(),
                self.query_one("#model-quant", Input).value.strip(),
                self.query_one("#model-revision", Input).value.strip(),
                self.query_one("#model-file", Select).value,
                self.query_one("#model-remote-projector", Select).value,
                self.query_one("#model-remote-mtp", Select).value,
                self.query_one("#model-target", Select).value,
                self.query_one("#model-verify-full", Checkbox).value)

    async def perform(self, id_: str) -> None:
        completed_model = None
        try:
            if id_ == "model-register":
                projector = self.query_one("#model-projector", Input).value.strip()
                mtp = self.query_one("#model-mtp", Input).value.strip()
                completed_model = await asyncio.to_thread(self.service.register, self.query_one("#model-path", Input).value,
                    **({"projector_path": projector} if projector else {}), **({"mtp_path": mtp} if mtp else {}))
            elif id_ == "model-list":
                self.install_plan = None
                self.query_one("#model-plan", Static).update("")
                self.query_one("#model-progress", Static).update("Checking repository and existing files; unchanged verified files are reused…")
                source, quant, revision, variant, projector, mtp, target, force_verify = self.install_selection()
                if source != self.listed_repo:
                    self.query_one("#model-file", Select).set_options([])
                    self.query_one("#model-file", Select).display = False
                    variant = Select.NULL
                self.listed_repo = source
                requested_selection = self.install_selection()
                try:
                    plan = await asyncio.to_thread(self.service.plan_install, source,
                        quant=variant if variant is not Select.NULL else (quant or None) if target == "new" else None,
                        revision=revision or None, projector=projector, mtp=mtp, model_id=None if target == "new" else target, force_verify=force_verify)
                except Exception as original:
                    try:
                        choices = await asyncio.to_thread(self.service.list_install_choices, source, revision or "main")
                    except Exception:
                        raise original
                    self.query_one("#model-file", Select).set_options([(safe(row["filename"]), row["filename"]) for row in choices["models"]])
                    self.query_one("#model-file", Select).display = len(choices["models"]) > 1
                    self.query_one("#model-remote-projector", Select).set_options(
                        [("Automatic", "auto"), ("Skip / keep existing", "none")]
                        + [(safe(row["filename"]), row["filename"]) for row in choices["projectors"]])
                    self.query_one("#model-remote-projector", Select).value = projector
                    self.query_one("#model-remote-mtp", Select).set_options(
                        [("Skip / keep existing", "none"), ("Automatic matching head", "auto")]
                        + [(safe(row["filename"]), row["filename"]) for row in choices.get("mtp_heads", [])])
                    self.query_one("#model-remote-mtp", Select).value = mtp
                    raise
                self.install_plan = plan
                self.preview_selection = requested_selection
                count = len(plan["files"])
                self.query_one("#model-plan", Static).update(safe(
                    f"{'Existing model (weights unchanged)' if target != 'new' else 'Model'}: {plan['filename']}\n"
                    f"Vision: {plan['projector_filename'] or 'no download'}\n"
                    f"MTP: {plan.get('mtp_filename') or 'no download'}\n"
                    f"Files to install: {count}\n"
                    f"Total: {plan['size_bytes'] / 2**30:.2f} GiB; verified in cache: {plan['cached_bytes'] / 2**30:.2f} GiB\n"
                    f"Save to: {plan['destination']}\nCommit: {plan['revision']}"))
                self.query_one("#model-progress", Static).update("Ready. Download and register installs this exact revision; no GPU allocation is started.")
            elif id_ == "model-download":
                plan = self.install_plan
                if not plan or self.preview_selection != self.install_selection():
                    self.install_plan = None
                    raise ValueError("The selection changed. Preview download again before installing.")
                self.query_one("#model-progress", Static).update(f"Installing the previewed model at commit {plan['revision'][:12]}…")
                completed_model = await asyncio.to_thread(self.service.execute_install, plan,
                    cancel=self.cancel_event, progress=self.download_progress)
        except Exception as exc:
            self.query_one("#model-progress", Static).update(safe(getattr(exc, "message", exc)))
        finally:
            self.busy = False
            for button in ("model-register", "model-list", "model-download"):
                self.query_one("#" + button, Button).disabled = False
            self.query_one("#model-download", Button).disabled = self.install_plan is None
            self.query_one("#model-cancel", Button).disabled = True
        if self.close_requested:
            self.dismiss(None)
        elif isinstance(completed_model, Path):
            self.query_one("#model-progress", Static).update(safe(f"Downloaded to: {completed_model}"))
        elif completed_model is not None:
            self.dismiss(data(completed_model))


class LauncherApp(App):
    """Returns a start/resume choice; root CLI submits only after this UI exits."""
    TITLE = "HPC LLM"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [Binding("ctrl+d", "quit", "Exit", priority=True),
                Binding("ctrl+q", "quit", "Exit", priority=True, show=False)]
    CSS = """
    Screen { align: center middle; }
    #launch { width: 94%; max-width: 120; height: auto; max-height: 100%; padding: 1 2; border: ascii $accent; }
    #launch-title { height: 1; text-wrap: nowrap; text-overflow: ellipsis; text-style: bold; color: $accent; margin-bottom: 1; }
    #launch-body { height: auto; max-height: 11; }
    #launch Label { height: auto; margin-top: 1; }
    #launch Static { height: auto; }
    #launch Button { margin-top: 0; }
    #launch .actions { height: 4; min-height: 4; }
    #launch .actions Button { margin-top: 1; }
    #launch SelectCurrent { height: 3; }
    #launch SelectCurrent Static#label { height: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    #launch-model-row { height: 3; }
    #launch-model { width: 1fr; }
    .session-row { height: 3; }
    .session-row Select { width: 1fr; }
    #launch .session-row Button { width: 12; min-width: 10; margin: 0 0 0 1; }
    #import-model { width: 14; min-width: 12; margin: 0 0 0 1; }
    #launch-resource-labels { height: 2; }
    #launch-resource-labels Label { width: 1fr; }
    #launch-resource-labels #launch-time-label { width: 23; margin: 1 0 0 1; }
    #launch-resource-row { height: 3; }
    #resources { width: 1fr; margin-top: 0; }
    #launch-walltime { width: 23; margin-left: 1; }
    #launch-error { color: $error; }
    #launch .help { color: $text-muted; }
    """ + PORTABLE_BORDERS_CSS

    def __init__(self, models, sessions, profile, model_service=None, inventory=None, error_message: str = "", profile_service=None):
        super().__init__()
        self.models = [data(m) for m in models]
        self.sessions = sorted([data(s) for s in sessions], key=lambda session: session.get("created_at", 0), reverse=True)
        self.resumable_sessions = [session for session in self.sessions if session_can_resume(session)]
        self.saved_sessions = [session for session in self.sessions if session_is_ended(session)]
        self.profile = data(profile)
        self.resources = dict(self.profile.get("resources", ResourceRequest().model_dump()))
        self.settings = InferenceSettings().model_dump()
        self.model_service = model_service
        self.inventory = inventory or {}
        self.error_message = error_message
        self.profile_service = profile_service

    def on_mount(self) -> None:
        self.limit_body_height(self.size.height)

    def on_resize(self, event: Resize) -> None:
        self.limit_body_height(event.size.height)

    def limit_body_height(self, rows: int) -> None:
        # Leave room for title, frame/padding, fixed actions, footer and margins.
        # Auto height then fits content; only short terminals need body scrolling.
        if self.query("#launch-body"):
            self.query_one("#launch-body").styles.max_height = max(1, rows - 13)

    @staticmethod
    def session_label(session: dict) -> str:
        state = session.get("backend_state", "unknown") if session.get("demo") else session.get("scheduler_state", "unknown")
        return safe(f"{session.get('model', {}).get('name', session['id'][:8])} · {state} · {session['id'][:8]}")

    def compose(self) -> ComposeResult:
        with Vertical(id="launch"):
            yield plain(f"HPC LLM · {self.profile.get('name', 'Site')} · {self.profile.get('scheduler', 'slurm').upper()}", id="launch-title")
            with VerticalScroll(id="launch-body"):
                if self.resumable_sessions:
                    yield Label("Resume running session (or wait for a queued job)")
                    with Horizontal(classes="session-row"):
                        yield Select([(self.session_label(s), s["id"]) for s in self.resumable_sessions], value=self.resumable_sessions[0]["id"], allow_blank=False, id="resume-session")
                        yield Button("Resume", id="resume", variant="primary")
                elif self.sessions:
                    yield plain("No running GPU session to resume.", classes="help")
                if self.saved_sessions:
                    yield Label("Restore saved chats (starts a new GPU allocation)")
                    with Horizontal(classes="session-row"):
                        yield Select([(self.session_label(s), s["id"]) for s in self.saved_sessions], value=self.saved_sessions[0]["id"], allow_blank=False, id="restore-session")
                        yield Button("Restore…", id="restart-saved")
                yield Label("1  Model")
                with Horizontal(id="launch-model-row"):
                    yield Select([(safe(m["name"]), m["id"]) for m in self.models], value=self.models[0]["id"] if self.models else Select.NULL, allow_blank=not bool(self.models), id="launch-model")
                    yield Button("Models…", id="import-model")
                with Collapsible(title="Model details", collapsed=True):
                    yield plain("", id="launch-model-info", classes="help")
                with Horizontal(id="launch-resource-labels"):
                    yield Label("2  Resources")
                    yield Label("Time limit (HH:MM:SS)", id="launch-time-label")
                with Horizontal(id="launch-resource-row"):
                    yield Button(self.resource_summary(), id="resources")
                    yield Input(self.resources["walltime"], id="launch-walltime", tooltip="For this new allocation only. Use HH:MM:SS or D-HH:MM:SS; the saved profile stays unchanged.")
                yield plain("No GPU job starts merely by opening this screen.", classes="help")
                yield plain(self.error_message, id="launch-error", classes="error")
            with Horizontal(classes="actions"):
                yield Button("Start session", variant="primary", id="start", disabled=not bool(self.models))
                if self.profile_service:
                    yield Button("Site setup…", id="site-setup")
                yield Button("Exit", id="launch-exit")
        yield Footer()

    def resource_summary(self) -> str:
        r = self.resources
        return f"{r['gpu_type']} × {r['gpu_count']} · {r['cpus']} CPUs · {r['memory_gb']} GB · {r['partition']}"

    def validate_launch_resources(self) -> bool:
        try:
            self.resources = ResourceRequest(**dict(self.resources, walltime=self.query_one("#launch-walltime", Input).value.strip())).model_dump()
        except ValueError:
            error = self.query_one("#launch-error", Static)
            error.update("Time limit must be positive: use HH:MM:SS or D-HH:MM:SS (for example 02:00:00).")
            self.query_one("#launch-walltime", Input).focus()
            error.scroll_visible(immediate=True)
            return False
        self.query_one("#launch-error", Static).update("")
        return True

    @on(Select.Changed, "#launch-model")
    def model_changed(self, event: Select.Changed) -> None:
        model = next((m for m in self.models if m["id"] == event.value), None)
        if model:
            defaults = {"context": 8192} if model.get("projector_path") else {}
            defaults.update(model.get("defaults") or {})
            self.settings = InferenceSettings(**defaults).model_dump()
            weights = f"{model.get('size_bytes', 0) / 2**30:.2f} GiB weights" if model.get("size_bytes") else "Weight size unknown"
            self.query_one("#launch-model-info", Static).update(safe(f"{model.get('quantization') or 'Quantization unknown'} · {weights} · Context {model.get('supported_context') or 'unknown'}\nVRAM also needs context/KV and runtime buffers; weights alone do not guarantee fit."))

    @on(Button.Pressed)
    def buttons(self, event: Button.Pressed) -> None:
        id_ = event.button.id
        if id_ == "launch-exit":
            self.exit(None)
        elif id_ == "resume":
            self.exit({"action": "resume", "session_id": self.query_one("#resume-session", Select).value})
        elif id_ == "restart-saved":
            if self.validate_launch_resources():
                self.push_screen(ConfirmDialog("Start a new GPU allocation for saved chats?", f"Time limit: {self.resources['walltime']}. Uses the saved model and conversations with the currently selected resources. The previous allocation must have ended; live sessions should use Resume. Nothing starts until you confirm.", "Start new"), self.restart_result)
        elif id_ == "start":
            model_id = self.query_one("#launch-model", Select).value
            if model_id is not Select.NULL and self.validate_launch_resources():
                self.exit({"action": "start", "model_id": model_id, "resources": self.resources, "settings": self.settings})
        elif id_ == "resources":
            resources = dict(self.resources, walltime=self.query_one("#launch-walltime", Input).value.strip())
            self.push_screen(ResourceDialog(resources, self.inventory, self.profile), self.resource_result)
        elif id_ == "site-setup":
            try:
                self.push_screen(SiteSetupDialog(self.profile_service, self.profile), self.site_result)
            except Exception as exc:
                self.query_one("#launch-error", Static).update(safe(getattr(exc, "message", str(exc))))
        elif id_ == "import-model":
            self.push_screen(ImportDialog(self.model_service), self.import_result)

    def resource_result(self, result: dict | None) -> None:
        if result:
            self.resources = result
            self.query_one("#resources", Button).label = self.resource_summary()
            self.query_one("#launch-walltime", Input).value = result["walltime"]

    def site_result(self, result: dict | None) -> None:
        if result:
            self.exit(result)

    def restart_result(self, yes: bool) -> None:
        if yes and self.validate_launch_resources():
            self.exit({"action": "restart", "session_id": self.query_one("#restore-session", Select).value, "resources": self.resources})

    def import_result(self, result: dict | None) -> None:
        if result:
            self.models = [data(m) for m in self.model_service.list()]
            selector = self.query_one("#launch-model", Select)
            selector.set_options([(safe(m["name"]), m["id"]) for m in self.models])
            selector.value = result["id"]
            self.query_one("#start", Button).disabled = False
