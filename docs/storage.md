# Storage and browser history

The GPU job serves the chat interface and runs inference. Your browser stores the native chat history. These are separate stores.

| Item | Location | What survives a stopped job |
|---|---|---|
| Downloaded model weights, vision projectors and MTP companions | The selected model cache; `cache/models` by default | Files remain until you remove them. |
| Registered local model and companion paths | The model registry under the configured state directory, normally `state/models/models.json` | Registrations remain; local files stay at their original paths. |
| Job configuration, status and logs | `STATE/sessions/SESSION_ID/`, including `manifest.json` and `logs/` | Session records remain. They do not contain a backup of native browser chats. |
| Native browser conversations and submitted attachments | IndexedDB in the browser profile that opened the chat | Usually survive a job stopping. Clearing site data, private browsing, browser eviction or changing browser profiles can remove access. |
| Native browser settings | Browser localStorage | Follow the same browser origin rules as history. |
| Explicitly saved cluster files | The exact path approved in an allowed workspace | Remain on cluster storage independently of the job. |
| Downloaded file or exported conversation | Your browser's download destination | The browser saves it on your computer unless you select a mounted cluster destination yourself. |

`STATE` means the configured application state directory. Keep it private: session access credentials and backend keys also live there. Do not publish it with a source archive or put it inside an allowed workspace.

## Which URLs share history?

The managed runtime is pinned to llama.cpp build **b11351**. Its native UI uses a database named `LlamaUi`, with `conversations` and `messages` tables. The name does not include the GPU session ID, node, port or URL path. Submitted attachment text and media data are stored with messages. Settings use `LlamaUi.*` localStorage keys.

Browser storage belongs to an **origin**: scheme, hostname and browser-facing port. Paths and URL fragments do not form a new origin.

| Reconnection | Expected visibility in the same browser profile |
|---|---|
| Another `/rnode/NODE/PORT/` path under the same `https://ondemand.example.edu` origin | The same native history database is visible. Conversations can appear across GPU sessions and models. |
| The same localhost URL and port after restarting a local SSH forward | The same history database is visible. |
| `localhost:8765` changed to `localhost:8766` | A different origin; the previous history is not visible there. |
| `localhost` changed to `127.0.0.1`, HTTP changed to HTTPS, or another OOD hostname | A different origin. |
| Another browser, profile or computer | Separate browser storage. |

A missing conversation after a URL change does not establish that it was deleted. Return to the original origin and browser profile, then export it. Reconnecting restores access to the existing allocation; it does not copy browser history between origins.

## Attachments and cluster files

Attaching a file in native chat sends its processed contents to the model and stores supported attachment data with the browser message. It does not create a named file in your cluster workspace. Large media, PDF processing and image analysis depend on the native UI and the loaded model; image inference needs a compatible vision model and projector.

Use **Files & tools** to browse cluster workspaces and approve an exact destination when saving generated text, code, `.txt` or `.csv` files. A successful save is a real cluster file. Its **Download** action makes a separate browser download. File-tool task activity lives in the GPU process's memory; use its transcript export to download a JSON copy before the job ends.

## Back up a conversation

Use the native sidebar's conversation export controls before clearing site data, changing computers or changing origins. The pinned UI exports one conversation as JSONL and multiple conversations as a ZIP containing JSONL files. These exports include the message records and their stored attachment data. Keep exports somewhere you back up and test an import before relying on them.

Existing terminal-chat snapshots remain in application state. This release does not import them into native browser history, or silently migrate or delete native conversations. Native history has no automatic server-side synchronisation or cluster backup.

## Source checks and runtime updates

These claims were checked against the pinned source, rather than inferred from the current upstream UI:

- [b11351 storage constants](https://github.com/ggml-org/llama.cpp/blob/b11351/tools/ui/src/lib/constants/storage.constants.ts): the fixed `LlamaUi` name and localStorage keys.
- [b11351 database service](https://github.com/ggml-org/llama.cpp/blob/b11351/tools/ui/src/lib/services/database.service.ts): Dexie conversations/messages persistence.
- [b11351 message types](https://github.com/ggml-org/llama.cpp/blob/b11351/tools/ui/src/lib/types/database.d.ts): attachment text and media fields in `DatabaseMessage.extra`.
- [b11351 conversation transfer service](https://github.com/ggml-org/llama.cpp/blob/b11351/tools/ui/src/lib/services/conversation-transfer.service.ts): JSONL/ZIP export and browser download.
- [MDN same-origin policy](https://developer.mozilla.org/en-US/docs/Web/Security/Defenses/Same-origin_policy): browser storage isolation by scheme, hostname and port.

When changing the runtime lock, inspect these files at the replacement tag. Check history and attachment retention, export/import, OOD path prefixes and the tools integration in a real browser before updating the documentation. A source inspection establishes the storage mechanism; it does not test a site's browser policies or backup arrangements.
