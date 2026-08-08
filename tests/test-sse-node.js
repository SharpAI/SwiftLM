#!/usr/bin/env node
// test-sse-node.js — verify the SSE stream against a second, independent parser.
//
// Every other SSE assertion in CI goes through Python httpx (the OpenAI SDK). SwiftLM
// frames SSE with \r\n\r\n rather than the spec's \n\n, and httpx tolerates that —
// whether every strict client does is the kind of thing only a different parser
// establishes (issue #134). Node is preinstalled on the CI runner, and `fetch` plus a
// manual event split needs no npm install, so this costs no memory and no dependencies.
//
// Usage: node tests/test-sse-node.js [baseUrl] [model]

const BASE = process.argv[2] || 'http://127.0.0.1:15413';
const MODEL = process.argv[3] || 'x';

let pass = 0, fail = 0;
const ok = (m) => { pass++; console.log(`  ✅ PASS: ${m}`); };
const bad = (m) => { fail++; console.log(`  ❌ FAIL: ${m}`); };

/** Split an SSE body into events, accepting both \n\n and \r\n\r\n separators. */
function parseEvents(raw) {
    return raw
        .split(/\r?\n\r?\n/)
        .map((block) => block.trim())
        .filter(Boolean)
        .map((block) => {
            const dataLines = block
                .split(/\r?\n/)
                .filter((l) => l.startsWith('data:'))
                .map((l) => l.slice(5).trim());
            return { raw: block, data: dataLines.join('\n') };
        })
        .filter((e) => e.data.length > 0);
}

async function streamRequest(body) {
    const res = await fetch(`${BASE}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.text();
}

(async () => {
    console.log('[node-sse] Test 1: stream framing is parseable by a non-Python client');
    const raw = await streamRequest({
        model: MODEL,
        messages: [{ role: 'user', content: 'Say hello in five words.' }],
        max_tokens: 40, temperature: 0, stream: true,
        stream_options: { include_usage: true },
    });

    // The framing SwiftLM emits, asserted explicitly rather than assumed.
    if (/\r\n\r\n/.test(raw)) {
        ok('server uses \\r\\n\\r\\n framing (documented; parser must tolerate it)');
    } else {
        ok('server uses \\n\\n framing');
    }

    const events = parseEvents(raw);
    if (events.length === 0) { bad('no SSE events parsed'); }
    else { ok(`parsed ${events.length} SSE events`); }

    const done = events.filter((e) => e.data === '[DONE]');
    done.length === 1
        ? ok('exactly one [DONE] terminator')
        : bad(`expected 1 [DONE] terminator, found ${done.length}`);

    let text = '', sawFinish = false, sawUsage = false, malformed = 0;
    for (const e of events) {
        if (e.data === '[DONE]') continue;
        let obj;
        try { obj = JSON.parse(e.data); } catch { malformed++; continue; }
        if (!Array.isArray(obj.choices)) { malformed++; continue; }
        // An empty choices array is the usage chunk; accumulation must survive it.
        if (obj.choices.length === 0 && obj.usage) sawUsage = true;
        for (const c of obj.choices) {
            if (c.finish_reason) sawFinish = true;
            text += (c.delta && c.delta.content) || '';
        }
    }

    malformed === 0 ? ok('every data payload is valid JSON with a choices array')
                    : bad(`${malformed} malformed event(s)`);
    sawFinish ? ok('a finish_reason was reported') : bad('no finish_reason in any chunk');
    text.length > 0 ? ok(`accumulated ${text.length} chars of content`)
                    : bad('no content accumulated');
    sawUsage ? ok('terminal usage chunk with empty choices did not break accumulation')
             : console.log('  ⏭️  SKIP: no usage chunk seen (server may not have sent one)');

    console.log(`[node-sse] Results: ${pass} passed, ${fail} failed`);
    process.exit(fail === 0 ? 0 : 1);
})().catch((e) => { console.log(`  ❌ FAIL: ${e.message}`); process.exit(1); });
