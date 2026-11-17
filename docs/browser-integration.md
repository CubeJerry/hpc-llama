# Native chat and file tools

The managed runtime is pinned to llama.cpp b11351. Its native UI remains the chat interface. HPC Llama adds a **Files & tools** disclosure containing a same-origin iframe, served by the authenticated entry point inside the GPU job. It does not rebuild or vendor the native UI.

The integration lives in:

- `src/hpc_llm/browser.py`: authenticated proxy, prefix handling, bounded identity/gzip HTML transformation and denial of native `/tools` execution routes.
- `src/hpc_llm/browser_login.py`: fragment-to-cookie login and an early browser initializer that removes private access fragments on repeat reconnects before native chat routing.
- `src/hpc_llm/browser_files.py`: HTML insertion, workspace access, validated tool schemas, approvals and bounded model execution loop.
- `src/hpc_llm/browser_files.html` and `browser_files.js`: visible file controls, preview/approval UI, task activity and transcript downloads.
- `src/hpc_llm/files.py`: reused directory/file safety and atomic publication helpers.

The panel uses `/_hpc/files` beneath the configured public prefix. Cookie authentication and origin checks apply before its routes are reached. It makes same-origin requests to the job's file executor. The inference API credential stays inside the Python process; the browser does not receive it.

## Manual file operations

Browse or refresh the selected workspace, open supported UTF-8 files, enter an exact destination, and paste text/code into the editor. **Save** requests a preview; approving it publishes the file. Directory creation follows the same preview/approval path. Overwrites include the previous/proposed-content diff. Preview tokens are short-lived and single-use, and publication checks that the destination has not changed since preview.

The panel reports success only after the filesystem operation completes. Downloads return the actual saved bytes. Neither browser attachments nor native chat responses are automatically saved into cluster workspaces.

## Model-driven operations

**Ask the model to work with files** starts a separate file task. Its context includes the task prompt and tool results, rather than reading the native UI's conversation or browser attachments. To use a native response, paste it into the editor or the file-task prompt yourself.

The server sends the loaded model the bounded `list_workspace`, `read_workspace`, `write_workspace` and `mkdir_workspace` schemas. It validates each requested tool and its arguments, executes permitted reads, and pauses writes for a visible approval. A successful execution result returns to the model as a tool-result message before the next completion request. The panel displays activity, failures and the final answer. `max_tool_calls` and `timeout_seconds` bound the loop; cancellation stops it. Model support for tool calling still depends on the loaded model and template.

`allow_writes: true` deliberately removes per-write approval for model tasks within the selected workspace. It does not expand the workspace boundary, enable shell execution, or remove the manual editor's preview. With the default `false`, model writes require approval too. Cancellation covers model/network/approval waits and leaves no background write running; it cannot forcibly interrupt a stalled operating-system/filesystem call.

Native llama.cpp `/tools` execution routes are blocked at the gateway. This avoids exposing a second executor outside the wrapper's workspace/approval policy. External native MCP integrations and the old terminal web-search tools are not part of this file workflow.

## Updating llama.cpp

Keep the runtime lock update and integration changes reviewable together. Inspect the replacement tag's UI/storage behavior and embedded-server HTML encoding, then run the configuration, connection, cookie-login, compute-web and file-panel checks. Run `check_browser_files.py --browser` where Chromium is available, followed by the [cluster smoke test](verification.md) with the shipped native UI.

Check native assets, model discovery, streaming and image bodies through the actual prefix; opening HTML alone is insufficient. Confirm that the disclosure appears without disturbing native chat, the iframe follows the prefix, writes stay scoped and approved, and native `/tools` cannot bypass the executor. Recheck [history/storage source references](storage.md#source-checks-and-runtime-updates) before changing retention claims. Do not silently delete or migrate browser history during an update.
