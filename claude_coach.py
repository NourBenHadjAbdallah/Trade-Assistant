"""
claude_coach.py - Claude as a coach backend (Ollama stays the default).

Two ways in, picked automatically:
  * API key   : ANTHROPIC_API_KEY in .env -> official Anthropic SDK (pay per use).
  * Subscription: no key -> the Claude Code CLI (`claude`), signed in with a Claude Pro/Max plan.
                 Install Claude Code and run `claude` once to sign in.

Both stream text through on_token(piece) so the GUI can show the answer as it is written.
"""
import json
import os
import shutil
import subprocess
import tempfile

import config  # noqa: F401  (loads .env before os.environ is read)

CLAUDE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", CLAUDE_MODELS[0])
# models that accept the server-side refusal fallback ("default" routing)
_FALLBACK_OK = {"claude-opus-5-5", "claude-sonnet-5-5"}


def cli_path():
    return shutil.which("claude")


def mode():
    """'api', 'subscription' or None (not set up)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    if cli_path():
        return "subscription"
    return None


def status_text():
    m = mode()
    if m == "api":
        return "Claude via API key (.env). Billed per use."
    if m == "subscription":
        return "Claude via your Claude subscription (Claude Code sign-in)."
    return ("Claude is not set up. Either install Claude Code and run `claude` once to sign in "
            "with your subscription, or add ANTHROPIC_API_KEY to .env.")


def ask(system, messages, model=None, on_token=None):
    """messages: [{"role": "user"|"assistant", "content": str}, ...] ending with the user turn."""
    m = mode()
    if m == "api":
        return ask_api(system, messages, model, on_token)
    if m == "subscription":
        return ask_cli(system, messages, model, on_token)
    raise RuntimeError(status_text())


def ask_api(system, messages, model=None, on_token=None):
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("The Claude SDK is not installed. Run: pip install anthropic")
    model = model or CLAUDE_MODEL
    extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"} if model in _FALLBACK_OK else {}
    parts = []
    try:
        client = anthropic.Anthropic()
        with client.beta.messages.stream(model=model, max_tokens=16000, system=system, messages=messages,
                                         cache_control={"type": "ephemeral"}, **extra) as stream:
            for piece in stream.text_stream:
                parts.append(piece)
                if on_token:
                    on_token(piece)
            final = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise RuntimeError("Claude rejected the API key. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.PermissionDeniedError:
        raise RuntimeError("This API key cannot use that model. Pick another Claude model.")
    except anthropic.RateLimitError:
        raise RuntimeError("Claude rate limit reached. Wait a minute and try again.")
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        raise RuntimeError("Cannot reach the Claude API. Check your internet connection.")
    if final.stop_reason == "refusal":
        raise RuntimeError("Claude declined to answer this one. Try rephrasing the question.")
    return "".join(parts).strip()


def _transcript(messages):
    """The CLI takes one prompt: earlier turns go in as a short transcript."""
    *past, last = messages
    if not past:
        return last["content"]
    lines = ["Conversation so far:"]
    lines += [f"{'User' if t['role'] == 'user' else 'Coach'}: {t['content']}" for t in past]
    return "\n\n".join(["\n".join(lines), "New question from the user:", last["content"]])


def ask_cli(system, messages, model=None, on_token=None, timeout=300):
    exe = cli_path()
    if not exe:
        raise RuntimeError(status_text())
    # drop API credentials so the CLI uses the subscription sign-in, not API billing
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    parts, result = [], None
    with tempfile.TemporaryDirectory() as tmp:      # empty cwd: no project CLAUDE.md, hooks or tools
        sys_path = os.path.join(tmp, "system.txt")
        with open(sys_path, "w", encoding="utf-8") as f:
            f.write(system)
        cmd = [exe, "-p", "--system-prompt-file", sys_path, "--tools", "", "--setting-sources", "",
               "--output-format", "stream-json", "--verbose", "--include-partial-messages",
               "--no-session-persistence", "--model", model or CLAUDE_MODEL]
        try:
            proc = subprocess.Popen(cmd, cwd=tmp, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as e:
            raise RuntimeError(f"Could not start Claude Code: {e}")
        proc.stdin.write(_transcript(messages))
        proc.stdin.close()
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") == "stream_event":
                e = ev.get("event", {})
                piece = e.get("delta", {}).get("text") if e.get("type") == "content_block_delta" else None
                if piece:
                    parts.append(piece)
                    if on_token:
                        on_token(piece)
            elif ev.get("type") == "result":
                result = ev
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            raise RuntimeError("Claude Code took too long to answer.")
    if result is None:
        raise RuntimeError("Claude Code returned no answer: " + (proc.stderr.read().strip()[:300] or "unknown error"))
    if result.get("is_error"):
        msg = str(result.get("result") or "unknown error")
        if "credit balance" in msg.lower():
            msg += (". Claude Code is signed in without a Claude subscription. Run `claude`, type /login "
                    "and choose your Claude Pro/Max account, or add ANTHROPIC_API_KEY to .env.")
        raise RuntimeError(f"Claude Code: {msg}")
    return ("".join(parts) or str(result.get("result", ""))).strip()
