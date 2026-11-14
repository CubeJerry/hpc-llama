"""Browser sessions that do not rely on Authorization surviving an OOD proxy."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets

from aiohttp import web


class BrowserLogin:
    def __init__(self, access_key: str, prefix: str, secure: bool):
        self.key = access_key
        self.root_path = prefix.rstrip('/') + '/'
        self.secure = secure
        self.cookie_name = 'hpc_chat_' + hashlib.sha256(access_key.encode()).hexdigest()[:16]
        self.cookie_value = hmac.new(access_key.encode(), b'browser-session-v1', hashlib.sha256).hexdigest()

    def authorized(self, request: web.Request) -> bool:
        value = request.cookies.get(self.cookie_name, '')
        return hmac.compare_digest(value.encode(), self.cookie_value.encode())

    async def gate(self, request: web.Request, path: str) -> web.Response | None:
        # The caller checks Origin / Sec-Fetch-Site before reaching this gate.
        if path == '/_hpc/login' and request.method == 'POST':
            try:
                try:
                    raw = await request.content.readexactly(4097)
                except asyncio.IncompleteReadError as exc:
                    raw = exc.partial
                if len(raw) > 4096:
                    raise web.HTTPRequestEntityTooLarge(max_size=4096, actual_size=len(raw))
                body = json.loads(raw)
                key = body.get('key', '') if isinstance(body, dict) else ''
                valid = isinstance(key, str) and hmac.compare_digest(key.encode(), self.key.encode())
            except (ValueError, UnicodeError):
                valid = False
            if not valid:
                return web.json_response({'error': 'This access key is incorrect. Open the complete private link printed by the launcher.'}, status=403, headers={'Cache-Control': 'no-store'})
            response = web.json_response({'ok': True}, headers={'Cache-Control': 'no-store'})
            response.set_cookie(self.cookie_name, self.cookie_value, path=self.root_path,
                                secure=self.secure, httponly=True, samesite='Strict')
            return response
        if self.authorized(request):
            return None
        if path == '/' and request.method == 'GET':
            nonce = secrets.token_urlsafe(24)
            page = LOGIN_PAGE.replace('__NONCE__', nonce).replace('__ROOT__', json.dumps(self.root_path))
            return web.Response(text=page, content_type='text/html', headers={
                'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer',
                'X-Content-Type-Options': 'nosniff',
                'Content-Security-Policy': f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'unsafe-inline'; connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'self'",
            })
        return web.json_response({'error': 'Open the private chat link from ./hpc-llm attach to sign in.'}, status=401,
                                 headers={'Cache-Control': 'no-store'})


LOGIN_PAGE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Open your chat</title><style>
body{font:16px system-ui,sans-serif;background:#16191f;color:#eef1f6;margin:0;padding:10vh 24px}
main{max-width:460px;margin:auto}h1{font-size:26px}p{line-height:1.5;color:#bfc8d8}
input,button{box-sizing:border-box;width:100%;padding:12px;margin:8px 0 16px;border-radius:6px;font:inherit}
input{background:#252b35;color:inherit;border:1px solid #6a778d}button{background:#bbd2ff;color:#132139;border:0;cursor:pointer}
#message{min-height:1.5em}code{overflow-wrap:anywhere}
</style></head><body><main><h1>Open your chat</h1>
<p id="message" role="status">Use the complete private link printed by the launcher to open your chat automatically.</p>
<form id="login"><label for="key">Access key (only if your terminal dropped the end of the link)</label>
<input id="key" type="password" autocomplete="off" required>
<button id="open" type="submit">Open chat</button></form>
<p>This is not your cluster password. Recover your private link using <code>./hpc-llm attach SESSION_ID</code>.</p>
</main><script nonce="__NONCE__">
const root = __ROOT__;
const form = document.getElementById('login');
const keyInput = document.getElementById('key');
const message = document.getElementById('message');
const button = document.getElementById('open');
const accessKey = new URLSearchParams(location.hash.slice(1)).get('access');
if (location.hash) history.replaceState(null, '', location.pathname + location.search);
async function signIn(key) {
  button.disabled = true;
  message.textContent = 'Opening your chat…';
  try {
    const response = await fetch(root + '_hpc/login', {
      method: 'POST', credentials: 'same-origin',
      headers: {'Content-Type': 'application/json'}, body: JSON.stringify({key})
    });
    if (!response.ok) throw new Error('Could not sign in. Reopen the complete private link or paste its access key below.');
    // Confirm the proxy preserved the session cookie before reloading.
    const check = await fetch(root + '_hpc/session', {credentials: 'same-origin'});
    if (!check.ok) throw new Error('The browser session cookie was blocked. Allow cookies for this Open OnDemand site and try again.');
    location.replace(root);
  } catch (error) {
    message.textContent = error.message;
    button.disabled = false;
  }
}
form.addEventListener('submit', event => {event.preventDefault(); signIn(keyInput.value.trim());});
if (accessKey) signIn(accessKey);
</script></body></html>'''
