"""Small typed tools for evidence, tables, and bounded workspace documents; no code execution."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .contracts import AppError, MAX_DOCUMENT_BYTES, MAX_DOCUMENT_ARGUMENT_CHARS


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchWeb(Arguments):
    query: str = Field(min_length=1, max_length=2000)


class FetchPage(Arguments):
    url: str = Field(min_length=1, max_length=4096)


class SearchFiles(Arguments):
    attachment_ids: list[str] = Field(min_length=1, max_length=20)
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(default=6, ge=1, le=12)


class ReadRange(Arguments):
    attachment_id: str = Field(min_length=1, max_length=128)
    locator: str = Field(min_length=1, max_length=512)


class Where(Arguments):
    column: str = Field(min_length=1, max_length=256)
    op: Literal["eq", "ne", "gt", "gte", "lt", "lte", "contains", "missing"]
    value: str | int | float | bool | None = None


class TableArgs(Arguments):
    column: str | None = Field(default=None, max_length=256)
    where: Where | None = None
    group_by: str | None = Field(default=None, max_length=256)
    limit: int = Field(default=100, ge=1, le=1000)
    sheet: str | None = Field(default=None, max_length=256)


class TableOperation(Arguments):
    attachment_id: str = Field(min_length=1, max_length=128)
    operation: Literal["count", "filter", "group", "summary"]
    arguments: TableArgs = Field(default_factory=TableArgs)

    @model_validator(mode="after")
    def required_columns(self):
        if self.operation == "filter" and self.arguments.where is None:
            raise ValueError("Filter requires a where condition")
        if self.operation == "group" and not self.arguments.group_by:
            raise ValueError("Group requires group_by")
        if self.operation == "summary" and not self.arguments.column:
            raise ValueError("Summary requires a numeric column")
        return self


class WriteWorkspaceFile(Arguments):
    destination: str = Field(min_length=1, max_length=4096)
    content: str = Field(max_length=MAX_DOCUMENT_BYTES)
    format: Literal["md", "txt"] = "md"

    @field_validator("content")
    @classmethod
    def bounded_utf8(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_DOCUMENT_BYTES:
            raise ValueError("Document exceeds 256 KiB")
        return value


SCHEMAS = {
    "write_workspace_file": (WriteWorkspaceFile, "Create a new Markdown (.md) or plain text (.txt) document in the selected workspace. Provide complete literal content, at most 256 KiB UTF-8. Parent directory must exist. Never overwrites files or executes content. Returns the actual saved path and checksum."),
    "web_search": (SearchWeb, "Search the selected public provider for the exact approved query. Results are snippets, not fetched pages."),
    "fetch_public_page": (FetchPage, "Read an approved public HTTP(S) HTML/text/PDF URL. Up to three public redirects may be followed; no scripts or forms."),
    "search_attached_files": (SearchFiles, "Search only explicitly attached immutable source snapshots. Returns excerpts with source IDs and coverage."),
    "read_attachment_range": (ReadRange, "Read a locator within an already attached snapshot. Does not accept filesystem paths."),
    "table_operation": (TableOperation, "Compute count, filter, group or numeric summary deterministically on an attached table's declared selection."),
}
WEB_TOOLS = frozenset({"web_search", "fetch_public_page"})


class ToolDispatcher:
    def __init__(self, files, web):
        self.files, self.web = files, web

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": name, "description": description,
                 "parameters": model.model_json_schema()}}
                for name, (model, description) in SCHEMAS.items()]

    def validate(self, name: str, args: dict | str) -> dict:
        if name not in SCHEMAS:
            raise AppError("permission", "That tool is not available. Only the listed source, table and workspace document tools can run.")
        argument_limit = MAX_DOCUMENT_ARGUMENT_CHARS if name == "write_workspace_file" else 12000
        if isinstance(args, str):
            if len(args) > argument_limit:
                raise AppError("validation", "Tool arguments exceed the permitted size.")
            try:
                args = json.loads(args, object_pairs_hook=_unique_keys, parse_constant=_reject_nonfinite)
            except (ValueError, TypeError, RecursionError):
                raise AppError("validation", "Tool arguments must be a valid JSON object without duplicate keys.") from None
        if not isinstance(args, dict):
            raise AppError("validation", "Tool arguments must be a JSON object.")
        try:
            if len(json.dumps(args, allow_nan=False)) > argument_limit:
                raise ValueError
            parsed = SCHEMAS[name][0].model_validate(args)
        except (ValidationError, ValueError, TypeError, RecursionError):
            raise AppError("validation", "The tool arguments do not match its allowed schema. Check required fields and types.") from None
        return parsed.model_dump(exclude_none=True)

    async def execute(self, name: str, args: dict, web_enabled: bool, provider: str = "ddgs") -> dict:
        """Called only after supervisor permission checks; never creates approvals.

        Web Off is enforced again here, including tools printed by a model.
        The supervisor owns per-turn time/count budgets and cancellation.
        """
        args = self.validate(name, args)
        if name in WEB_TOOLS:
            if not web_enabled:
                raise AppError("permission", "Web is Off. Enable it and approve the exact query or URL first.")
            if name == "web_search":
                sources = await self.web.search(args["query"], provider=provider)
            else:
                sources = [await self.web.fetch(args["url"])]
            return {"sources": [source.model_dump() for source in sources], "count": len(sources)}
        if name == "write_workspace_file":
            artifact = await asyncio.to_thread(self.files.write_document, **args)
            return {"artifact": artifact, "sources": []}
        ids = args.get("attachment_ids", [args.get("attachment_id")])
        if any(not isinstance(id_, str) or id_ not in self.files.attachments for id_ in ids):
            raise AppError("permission", "Tools can read only files already attached by the user.")
        if name == "search_attached_files":
            sources = await asyncio.to_thread(self.files.search, args["attachment_ids"], args["query"], args["limit"])
        elif name == "read_attachment_range":
            sources = await asyncio.to_thread(self.files.read_range, args["attachment_id"], args["locator"])
        else:
            result = await asyncio.to_thread(self.files.table, args["attachment_id"], args["operation"], args["arguments"])
            return {"result": result, "sources": []}
        return {"sources": [source.model_dump() for source in sources], "count": len(sources)}


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise ValueError("nonfinite value")
