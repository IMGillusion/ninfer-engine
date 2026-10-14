"""YaRN context-extension needle test against the local ninfer-serve endpoint.

Builds a filler context of a requested prompt size, hides one needle sentence at
~99% depth, and checks whether the model recalls it. Usage:
  python needle_yarn.py 4000
  python needle_yarn.py 150000
  python needle_yarn.py 335000
"""
import json
import sys
import time
import urllib.request

ENDPOINT = "http://127.0.0.1:9999/v1/chat/completions"
NEEDLE = "The emergency override code for the reactor subsystem is QT-4471-NC."
QUESTION = (
    "What is the emergency override code for the reactor subsystem? "
    "Reply with only the code itself and nothing else."
)
FILLER = (
    "The maintenance handbook describes quarterly inspection routines for the cooling "
    "loops. Technicians record flow rates, inlet temperatures, and bearing vibration "
    "spectra in the station log. Any deviation beyond two percent of the nominal "
    "envelope must be escalated to the duty engineer before the next shift begins. "
    "Filters are replaced on a fixed schedule and the spent elements are weighed so "
    "that particle accumulation can be tracked across seasons. Spare gaskets are kept "
    "in the sealed storage room on the lower level. "
)
CHARS_PER_TOKEN = 3.7


def call(messages, max_tokens=512, timeout=1800):
    body = json.dumps(
        {"model": "qwen3.8-27b", "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    ).encode()
    req = urllib.request.Request(
        ENDPOINT, data=body, headers={"Content-Type": "application/json"}
    )
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    result["wall_s"] = round(time.time() - started, 2)
    return result


def run(target_tokens):
    text = FILLER * max(1, int(target_tokens * CHARS_PER_TOKEN / len(FILLER)))
    # Calibrate once against the server tokenizer, then pad to within ~1%.
    probe = call([{"role": "user", "content": text}], max_tokens=1)
    per_char = probe["usage"]["prompt_tokens"] / len(text)
    while len(text) * per_char < target_tokens * 0.995:
        text += FILLER
    cut = int(len(text) * 0.99)
    text = text[:cut] + "\n" + NEEDLE + "\n" + text[cut:]
    result = call([{"role": "user", "content": text + "\n\n" + QUESTION}])
    answer = result["choices"][0]["message"]["content"]
    print(json.dumps({
        "target_prompt_tokens": target_tokens,
        "prompt_tokens": result["usage"]["prompt_tokens"],
        "needle_depth_tokens": int(result["usage"]["prompt_tokens"] * 0.99),
        "recall": "QT-4471-NC" in answer,
        "wall_s": result["wall_s"],
        "answer_tail": answer[-300:],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    run(int(sys.argv[1]))
