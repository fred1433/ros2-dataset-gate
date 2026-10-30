"""Command line: ``dsgate demo`` and ``dsgate readme --check|--write``."""

from __future__ import annotations

import sys
from pathlib import Path

START, END = "<!-- demo-output:start -->", "<!-- demo-output:end -->"


def readme(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="dsgate readme")
    ap.add_argument("--block", default="build/demo/README_block.md")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--provenance", default="", help="line written above the block (--write only)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    text = Path(args.readme).read_text()
    head, rest = text.split(START, 1)
    current, tail = rest.split(END, 1)
    block = Path(args.block).read_text()
    if args.check:
        body = current.split("```text", 1)
        current_block = "```text" + body[1] if len(body) == 2 else ""
        if current_block.strip() != block.strip():
            print("README block differs from this run's demo output", file=sys.stderr)
            return 1
        print("README block matches this run's demo output")
        return 0
    prov = (args.provenance + "\n\n") if args.provenance else ""
    Path(args.readme).write_text(f"{head}{START}\n{prov}{block}{END}{tail}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("demo", "readme"):
        print("usage: dsgate demo [--out DIR] [--train sagemaker-local|process|skip] | dsgate readme --check|--write")
        return 2
    if argv[0] == "demo":
        from .demo import main as demo_main

        return demo_main(argv[1:])
    return readme(argv[1:])


if __name__ == "__main__":
    sys.exit(main())
