#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Upload Anthropic Batch JSONL files for Claude 3.5 Haiku
=======================================================

This script uploads JSONL batch files (built by build_anthropic_haiku35_batches.py)
to Anthropic's Messages Batch API. It handles:

- Upload file to Anthropic Files API
- Create a messages batch job pointing at that file
- Optional status polling for submitted jobs

Environment
- ANTHROPIC_API_KEY: required
- ANTHROPIC_API_BASE (optional, default https://api.anthropic.com)
- ANTHROPIC_VERSION (optional, default 2023-06-01)

Usage examples
- Submit all JSONL in a folder:
  python upload_anthropic_haiku35_batches.py submit --glob "batches/anthropic_haiku35/*.jsonl"

- Check status for known batch IDs:
  python upload_anthropic_haiku35_batches.py status --ids <id1,id2,id3>

Notes
- This script uses the documented Anthropic headers (x-api-key, anthropic-version).
- The JSONL format should match what build_anthropic_haiku35_batches.py creates:
  one JSON object per line with custom_id and params fields.
"""

import os
import sys
import time
import json
import glob
import argparse
from typing import Optional, List, Dict, Any

import requests

# Load .env if present so keys in .env are available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    try:
        v = os.getenv(name)
        return v if v not in (None, "") else default
    except Exception:
        return default


def _headers(api_key: str, version: str) -> Dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": version,
        "content-type": "application/json",
    }


def upload_file(api_base: str, api_key: str, version: str, path: str) -> str:
    url = f"{api_base.rstrip('/')}/v1/files"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": version,
    }
    files = {
        "file": (os.path.basename(path), open(path, "rb"), "application/jsonl"),
    }
    data = {
        # Purpose is advisory; Anthropic may accept without it
        "purpose": (None, "batch"),
    }
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=300)
    resp.raise_for_status()
    obj = resp.json()
    file_id = obj.get("id") or obj.get("file_id")
    if not file_id:
        raise RuntimeError(f"Upload succeeded but no file id found in response: {obj}")
    return file_id


def create_messages_batch(api_base: str, api_key: str, version: str, input_file_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/messages/batches"
    payload = {
        "input_file_id": input_file_id,
    }
    if metadata:
        payload["metadata"] = metadata
    resp = requests.post(url, headers=_headers(api_key, version), json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def get_messages_batch(api_base: str, api_key: str, version: str, batch_id: str) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/messages/batches/{batch_id}"
    resp = requests.get(url, headers=_headers(api_key, version), timeout=60)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="Upload Anthropic JSONL batch files and manage batch jobs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit", help="Upload JSONL files and create batch jobs")
    p_submit.add_argument("--glob", required=True, help="Glob for JSONL files, e.g. batches/anthropic_haiku35/*.jsonl")
    p_submit.add_argument("--meta", default=None, help="Optional JSON metadata string to attach to the batch (e.g. '{\"run\":\"haiku35\"}')")

    # Inline submission path (no Files API; sends requests directly in body)
    p_submit_inline = sub.add_parser("submit-inline", help="Create batch jobs by sending requests inline (no file upload)")
    p_submit_inline.add_argument("--glob", required=True, help="Glob for JSONL files to read lines from")
    p_submit_inline.add_argument("--max-requests", type=int, default=3000, help="Max requests per batch job (default 3000)")
    p_submit_inline.add_argument("--meta", default=None, help="Optional JSON metadata string for the batch")

    p_status = sub.add_parser("status", help="Check status for one or more batch IDs")
    p_status.add_argument("--ids", required=True, help="Comma-separated batch IDs")
    p_status.add_argument("--watch", action="store_true", help="Poll until all are terminal (succeeded/failed/cancelled)")
    p_status.add_argument("--interval", type=int, default=10, help="Polling interval seconds (default 10)")

    args = parser.parse_args()

    # Accept multiple env var names for convenience
    api_key = _env("ANTHROPIC_API_KEY") or _env("CLAUDE_API_KEY") or _env("claude_api_key")
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY or CLAUDE_API_KEY.")
        sys.exit(1)
    api_base = _env("ANTHROPIC_API_BASE", "https://api.anthropic.com")
    version = _env("ANTHROPIC_VERSION", "2023-06-01")

    if args.cmd == "submit":
        meta = None
        if args.meta:
            try:
                meta = json.loads(args.meta)
            except Exception as e:
                print(f"Warning: Could not parse --meta JSON: {e}. Ignoring.")
                meta = None

        paths = sorted(glob.glob(args.glob))
        if not paths:
            print(f"No files matched glob: {args.glob}")
            sys.exit(1)

        results = []
        for path in paths:
            try:
                print(f"Uploading file: {path}")
                fid = upload_file(api_base, api_key, version, path)
                print(f"  -> file_id: {fid}")
                batch = create_messages_batch(api_base, api_key, version, fid, meta)
                print(f"  -> batch: {batch.get('id') or batch}")
                results.append({"path": path, "file_id": fid, "batch": batch})
            except requests.HTTPError as he:
                print(f"HTTP error for {path}: {he}\n{getattr(he, 'response', None) and he.response.text}")
            except Exception as e:
                print(f"Error for {path}: {e}")

        # Summary
        print("\nSubmission summary:")
        for r in results:
            bid = r["batch"].get("id") if isinstance(r.get("batch"), dict) else None
            print(f"- {r['path']} -> file_id={r['file_id']} batch_id={bid}")
        return

    if args.cmd == "submit-inline":
        meta = None
        if args.meta:
            try:
                meta = json.loads(args.meta)
            except Exception as e:
                print(f"Warning: Could not parse --meta JSON: {e}. Ignoring.")
                meta = None

        paths = sorted(glob.glob(args.glob))
        if not paths:
            print(f"No files matched glob: {args.glob}")
            sys.exit(1)

        def _post_inline(reqs: list[dict]) -> dict:
            url = f"{api_base.rstrip('/')}/v1/messages/batches"
            payload = {"requests": reqs}
            if meta:
                payload["metadata"] = meta
            r = requests.post(url, headers=_headers(api_key, version), json=payload, timeout=300)
            r.raise_for_status()
            return r.json()

        for path in paths:
            print(f"Reading: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                lines = [json.loads(ln) for ln in f if ln.strip()]
            # Each line already has {custom_id, params}
            maxn = max(1, int(getattr(args, 'max_requests', 3000)))
            start = 0
            part = 1
            while start < len(lines):
                chunk = lines[start:start+maxn]
                try:
                    resp = _post_inline(chunk)
                    print(f"  -> chunk {part}: batch_id={resp.get('id')} status={resp.get('status')}")
                except requests.HTTPError as he:
                    print(f"HTTP error for chunk {part}: {he}\n{getattr(he, 'response', None) and he.response.text}")
                    break
                except Exception as e:
                    print(f"Error for chunk {part}: {e}")
                    break
                start += len(chunk)
                part += 1
        return

    if args.cmd == "status":
        ids = [s.strip() for s in args.ids.split(",") if s.strip()]
        if not ids:
            print("No batch IDs provided.")
            sys.exit(1)
        def _show_once():
            for bid in ids:
                try:
                    info = get_messages_batch(api_base, api_key, version, bid)
                    state = info.get("status") or info.get("state")
                    out_fid = info.get("output_file_id") or info.get("result_file_id")
                    print(f"{bid}: status={state} output_file_id={out_fid}")
                except requests.HTTPError as he:
                    print(f"HTTP error for {bid}: {he}\n{getattr(he, 'response', None) and he.response.text}")
                except Exception as e:
                    print(f"Error for {bid}: {e}")

        if not args.watch:
            _show_once()
            return
        # Watch mode
        terminal = {"succeeded", "failed", "cancelled", "completed", "complete"}
        while True:
            _show_once()
            # crude break condition: stop when all are terminal
            all_done = True
            for bid in ids:
                try:
                    info = get_messages_batch(api_base, api_key, version, bid)
                    state = str(info.get("status") or info.get("state") or "").lower()
                    if state not in terminal:
                        all_done = False
                        break
                except Exception:
                    all_done = False
                    break
            if all_done:
                break
            time.sleep(max(1, int(args.interval)))


if __name__ == "__main__":
    main()
