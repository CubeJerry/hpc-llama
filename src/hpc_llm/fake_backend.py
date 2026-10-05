"""Deterministic HTTP inference fixture. Used only by explicitly marked DEMO sessions."""
from __future__ import annotations
import argparse
import asyncio
import base64
import hashlib
import json
import os
import sys
from pathlib import Path
import re
from aiohttp import web


def create_app(key: str, alias: str, context: int, fixture_mode: str = '', vision: bool = False) -> web.Application:
    @web.middleware
    async def authenticate(request, handler):
        if fixture_mode != 'no-auth' and request.headers.get('Authorization') != 'Bearer ' + key:
            return web.json_response({'error': 'Unauthorized'}, status=401)
        return await handler(request)
    app = web.Application(middlewares=[authenticate], client_max_size=4 * 1024 * 1024)
    async def models(request):
        if fixture_mode == 'loading':
            return web.json_response({'error':'Loading model'}, status=503)
        return web.json_response({'data': [{'id': 'wrong-session' if fixture_mode == 'wrong-alias' else alias}]})
    async def props(request):
        template = '{% if enable_thinking %}reason{% endif %}'
        if fixture_mode == 'reasoning-effort':
            template += "{% set resolved_reasoning_effort = reasoning_effort | default('xhigh') %}{% if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}{{ raise_exception('invalid effort') }}{% endif %}"
        elif fixture_mode == 'unknown-template':
            template = '{{ messages }}'
        return web.json_response({'default_generation_settings': {'n_ctx': context, 'params': {}}, 'total_slots': 1,
            'chat_template': template,
            'chat_template_caps': {'supports_tool_calls': True}, 'modalities': {'vision': vision}, 'build_info': 'DEMO fake-http-v1'})
    async def chat(request):
        payload = await request.json()
        template_kwargs = payload.get('chat_template_kwargs', {})
        if fixture_mode == 'reasoning-effort' and template_kwargs.get('reasoning_effort', 'xhigh') not in ('low', 'medium', 'xhigh'):
            return web.json_response({'error': 'Invalid reasoning effort'}, status=400)
        messages = payload.get('messages', [])
        content = messages[-1].get('content', '') if messages else ''
        image_receipts = []
        if not isinstance(content, str):
            parts = content
            content = ' '.join(part.get('text', '') for part in parts if part.get('type') == 'text')
            for part in parts:
                if part.get('type') != 'image_url':
                    continue
                if not vision:
                    return web.json_response({'error': 'Fixture vision is disabled'}, status=400)
                url = part.get('image_url', {}).get('url', '')
                try:
                    media, encoded = url.split(',', 1)
                    if media not in ('data:image/png;base64', 'data:image/jpeg;base64', 'data:image/webp;base64'):
                        raise ValueError
                    raw = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError):
                    return web.json_response({'error': 'Fixture expects inline image bytes'}, status=400)
                image_receipts.append(f'DEMO received image bytes: {len(raw)}; SHA256 {hashlib.sha256(raw).hexdigest()}')
        if '[demo-error]' in content:
            return web.json_response({'error': {'message': 'Synthetic failure'}}, status=500)
        if '[demo-oom]' in content:
            return web.json_response({'error': {'message': 'out of memory'}}, status=503)
        response = web.StreamResponse(headers={'Content-Type': 'text/event-stream'})
        await response.prepare(request)
        async def send(delta=None, finish=None):
            await response.write(('data: ' + json.dumps({'choices': [{'delta': delta or {}, 'finish_reason': finish}]}) + '\n\n').encode())
        try:
            think = payload.get('chat_template_kwargs', {}).get('enable_thinking', True)
            if think:
                await send({'reasoning_content': 'DEMO: checking the supplied context. '})
            if '[demo-reasoning-only]' in content:
                await send(finish='length')
            elif ('[demo-write-file]' in content or '[demo-write-file-long]' in content) and payload.get('tools') and messages[-1].get('role') != 'tool':
                body = '# Generated document\n\nA deterministic document created by the real workspace tool.\n\n'
                if '[demo-write-file-long]' in content:
                    body += ('## Synthetic section\n\nThis is controlled fixture content for document transport.\n\n' * 1000)
                arguments = json.dumps({'destination': 'generated-document.md', 'content': body})
                # Split both the name and arguments, including arguments before
                # completion of the name, to exercise the real streaming parser.
                await send({'tool_calls': [{'index': 0, 'id': 'demo_document_1', 'type': 'function', 'function': {'name': 'write_workspace_', 'arguments': arguments[:32768]}}]})
                await send({'tool_calls': [{'index': 0, 'function': {'name': 'file', 'arguments': arguments[32768:65536]}}]})
                for offset in range(65536, len(arguments), 32768):
                    await send({'tool_calls': [{'index': 0, 'function': {'arguments': arguments[offset:offset + 32768]}}]})
                await send(finish='tool_calls')
            elif '[demo-tool]' in content and payload.get('tools') and messages[-1].get('role') != 'tool':
                # Exercise the real fragmented streamed tool protocol.
                await send({'tool_calls': [{'index': 0, 'id': 'demo_call_1', 'type': 'function', 'function': {'name': 'web_search', 'arguments': '{"query":'}}]})
                await send({'tool_calls': [{'index': 0, 'function': {'arguments': '"synthetic research"}'}}]})
                await send(finish='tool_calls')
            else:
                sources = sorted(set(re.findall(r'\[(?:W|F)\d+\]', json.dumps(messages))))
                user_text = content.split('\n\nATTACHMENT COVERAGE', 1)[0].split('\n\nSOURCE DATA', 1)[0]
                document_result = messages[-1].get('role') == 'tool' and messages[-1].get('tool_call_id') == 'demo_document_1'
                if document_result:
                    result = json.loads(content)
                    if result.get('error'):
                        answer = 'Could not create the document: ' + str(result['error'])
                    else:
                        artifact = result.get('artifact', result)
                        answer = 'Created document: ' + str(artifact.get('path', 'generated-document.md'))
                else:
                    answer = 'DEMO response: ' + user_text[:180].replace('[demo-slow]', '').strip()
                if '[demo-template-kwargs]' in content:
                    answer += '\nDEMO template kwargs: ' + json.dumps(template_kwargs, sort_keys=True)
                if image_receipts:
                    answer += '\n' + '\n'.join(image_receipts)
                if sources:
                    answer += '\nEvidence is available from ' + ' '.join(sources[:4]) + '.'
                if not document_result:
                    answer += '\nThis deterministic fixture exercises local streaming; it is not model inference.'
                delay = 0.06 if '[demo-slow]' in content else 0.012
                for fragment in re.findall(r'.{1,12}', answer, re.DOTALL):
                    await asyncio.sleep(delay)
                    await send({'content': fragment})
                await send(finish='stop')
            await response.write(b'data: [DONE]\n\n')
        except (ConnectionError, asyncio.CancelledError):
            pass
        return response
    app.router.add_get('/v1/models', models)
    app.router.add_get('/props', props)
    app.router.add_get('/health', models)
    app.router.add_post('/v1/chat/completions', chat)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--version', action='version', version='DEMO fake-http-v1')
    parser.add_argument('--key-file', '--api-key-file', dest='key_file', type=Path, required=True)
    parser.add_argument('--alias', required=True)
    parser.add_argument('--context', '--ctx-size', dest='context', type=int, required=True)
    parser.add_argument('--mmproj')
    for flag in ('--host', '--model', '--parallel', '--threads', '--threads-batch', '--gpu-layers', '--batch-size', '--ubatch-size', '--cache-type-k', '--cache-type-v', '--threads-http', '--log-verbosity', '--temp', '--top-p', '--top-k', '--min-p', '--seed', '--repeat-penalty', '--presence-penalty', '--frequency-penalty'):
        parser.add_argument(flag)
    parser.add_argument('--flash-attn', choices=['on','off','auto'])
    for flag in ('--jinja', '--no-webui', '--no-slots', '--offline', '--no-context-shift'):
        parser.add_argument(flag, action='store_true')
    args = parser.parse_args()
    fixture_mode = os.environ.get('HPC_LLM_FAKE_MODE', '')
    if fixture_mode in ('startup-fail', 'oom'):
        print('CUDA error: out of memory' if fixture_mode == 'oom' else 'Synthetic startup failure', file=sys.stderr, flush=True)
        raise SystemExit(2)
    web.run_app(create_app(args.key_file.read_text().strip(), args.alias, args.context, fixture_mode, bool(args.mmproj)), host='127.0.0.1', port=args.port,
                access_log=None, print=None)

if __name__ == '__main__':
    main()
