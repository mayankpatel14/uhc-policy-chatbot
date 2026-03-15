#!/usr/bin/env python3
"""CLI interface for the UHC policy chatbot (local dev with Ollama)."""

import argparse
import sys
import time

from chatbot.config import (
    OLLAMA_MODEL,
    RETRIEVAL_TOP_K,
    MAX_HISTORY_TURNS,
)
from chatbot.retriever import PolicyRetriever
from chatbot.llm import OllamaClient, OllamaError
from chatbot.prompts import format_context, build_messages

# -- ANSI colors --------------------------------------------------------------
DIM = "\033[2m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    print(f"""
{BOLD}{'=' * 64}
  UHC Medical Policy Chatbot
  Model: Phi-3.5 Mini via Ollama  |  Retrieval: MedEmbed + Qdrant
{'=' * 64}{RESET}

{DIM}Commands:
  /clear   — reset conversation history
  /debug   — show retrieved chunks for the last query
  /quit    — exit{RESET}
""")


def print_sources(chunks):
    """Print a compact list of sources used."""
    if not chunks:
        return
    seen = set()
    print(f"\n{DIM}Sources:", end="")
    for c in chunks:
        key = f"{c.policy_name}/{c.section}"
        if key not in seen:
            seen.add(key)
            print(f"\n  [{c.score:.2f}] {c.policy_name} — {c.section}", end="")
    print(RESET)


def print_debug(chunks):
    """Print full debug info for retrieved chunks."""
    print(f"\n{YELLOW}{'─' * 64}")
    print(f"  DEBUG: {len(chunks)} chunks retrieved")
    print(f"{'─' * 64}{RESET}")
    for i, c in enumerate(chunks, 1):
        print(f"\n{YELLOW}  [{i}] score={c.score:.4f}  {c.policy_name} / {c.section}{RESET}")
        print(f"{DIM}  Plan: {c.plan_type}  Pages: {c.page_start}-{c.page_end}")
        preview = c.text[:300].replace("\n", " ")
        print(f"  {preview}...{RESET}")
    print()


def main():
    parser = argparse.ArgumentParser(description="UHC Policy Chatbot CLI")
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL)
    args = parser.parse_args()

    print_banner()

    # -- Check Ollama ---------------------------------------------------------
    llm = OllamaClient(model=args.model)
    err = llm.check_ready()
    if err:
        print(f"{RED}ERROR: {err}{RESET}")
        sys.exit(1)
    print(f"{DIM}Ollama ready ({args.model}){RESET}")

    # -- Init retriever -------------------------------------------------------
    retriever = PolicyRetriever()
    retriever.init(status_callback=lambda msg: print(f"{DIM}{msg}{RESET}"))
    print()

    # -- REPL -----------------------------------------------------------------
    history: list[dict] = []
    last_chunks = []
    debug_mode = False

    while True:
        try:
            query = input(f"{CYAN}{BOLD}> {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not query:
            continue

        # -- Commands ---------------------------------------------------------
        if query.lower() == "/quit":
            print(f"{DIM}Goodbye.{RESET}")
            break
        if query.lower() == "/clear":
            history.clear()
            last_chunks.clear()
            print(f"{DIM}History cleared.{RESET}\n")
            continue
        if query.lower() == "/debug":
            if last_chunks:
                print_debug(last_chunks)
            else:
                print(f"{DIM}No chunks retrieved yet.{RESET}\n")
            continue

        # -- Retrieve ---------------------------------------------------------
        t_start = time.perf_counter()
        try:
            chunks = retriever.retrieve(query, top_k=args.top_k)
        except RuntimeError as e:
            print(f"{RED}Retrieval error: {e}{RESET}\n")
            continue
        t_retrieval = time.perf_counter()

        last_chunks = chunks
        context = format_context(chunks)

        # -- Build messages and stream ----------------------------------------
        messages = build_messages(query, context, history=history)

        print(f"\n{GREEN}", end="", flush=True)
        full_response = []
        token_count = 0
        t_first_token = None
        try:
            for token in llm.chat_stream(messages):
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                print(token, end="", flush=True)
                full_response.append(token)
                token_count += 1
        except OllamaError as e:
            print(f"{RESET}\n{RED}LLM error: {e}{RESET}\n")
            continue
        t_done = time.perf_counter()

        print(RESET)

        # -- Sources ----------------------------------------------------------
        print_sources(chunks)

        # -- Latency ----------------------------------------------------------
        retrieval_ms = (t_retrieval - t_start) * 1000
        first_tok_ms = ((t_first_token or t_done) - t_retrieval) * 1000
        gen_ms = (t_done - (t_first_token or t_retrieval)) * 1000
        total_ms = (t_done - t_start) * 1000
        tok_per_s = token_count / (gen_ms / 1000) if gen_ms > 0 else 0

        print(f"\n{DIM}{'─' * 48}")
        print(f"  Retrieval:    {retrieval_ms:7.0f} ms")
        print(f"  First token:  {first_tok_ms:7.0f} ms")
        print(f"  Generation:   {gen_ms:7.0f} ms  ({token_count} tok, {tok_per_s:.1f} tok/s)")
        print(f"  Total:        {total_ms:7.0f} ms")
        print(f"{'─' * 48}{RESET}")
        print()

        # -- Update history ---------------------------------------------------
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": "".join(full_response)})

        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-(MAX_HISTORY_TURNS * 2):]


if __name__ == "__main__":
    main()
