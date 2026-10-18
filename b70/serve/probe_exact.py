"""probe_exact.py: bit-exactness probe between two server arms (e.g. FTX_SRC=ckpt vs FTX_SRC=usm).
  probe_exact.py run URL OUT.json      fresh server, C1: fixed short chat prompts, greedy, thinking off, natural end
                                       (no output cap), token ids + top-5 logprobs of every generated token
  probe_exact.py compare A.json B.json identical texts, token ids and logprobs? prints the first divergence
Run each arm on a FRESH server with the same flags and the same request sequence (the device expert cache and the
FTX split follow the call history), and check the FTX stats line for spin_timeouts == 0 in both."""
import json, sys, urllib.request

PROMPTS = [
    "List the first ten prime numbers, separated by commas.",
    "Translate to French: The cat sleeps on the warm windowsill while the rain falls outside.",
    "Write a haiku about a lighthouse in winter.",
    "What is 17 * 23? Show the multiplication step by step, then give the result.",
    "Name three differences between TCP and UDP, one short sentence each.",
    "Explain in two sentences why the sky is blue.",
]


def run(url, out):
    res = []
    for p in PROMPTS:
        body = {"model": "glm53-flash", "messages": [{"role": "user", "content": p}], "temperature": 0, "top_p": 1,
                "logprobs": True, "top_logprobs": 5, "chat_template_kwargs": {"enable_thinking": False}, "seed": 0}
        req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            j = json.load(r)
        c = j["choices"][0]
        lp = [{"t": t["token"], "lp": t["logprob"], "top": [(x["token"], x["logprob"]) for x in t.get("top_logprobs") or []]}
              for t in (c.get("logprobs") or {}).get("content") or []]
        res.append({"prompt": p, "text": c["message"].get("content"), "finish": c["finish_reason"],
                    "usage": j.get("usage"), "logprobs": lp})
        print(f"{len(lp):4d} tokens, {c['finish_reason']}: {(c['message'].get('content') or '')[:80]!r}", flush=True)
    json.dump(res, open(out, "w"), indent=1)


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    bad = 0
    ntok = 0
    for i, (x, y) in enumerate(zip(A, B)):
        la, lb = x["logprobs"], y["logprobs"]
        ntok += len(la)
        if x["text"] == y["text"] and la == lb:
            continue
        bad += 1
        k = next((k for k in range(min(len(la), len(lb))) if la[k] != lb[k]), min(len(la), len(lb)))
        print(f"prompt {i}: diverges at token {k} of {len(la)}/{len(lb)}: "
              f"{la[k] if k < len(la) else None} vs {lb[k] if k < len(lb) else None}")
    print(f"{len(A)} prompts, {ntok} tokens: {'IDENTICAL (texts, token ids, top-5 logprobs)' if not bad and len(A) == len(B) else f'{bad} differ'}")
    return 0 if not bad and len(A) == len(B) else 1


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
