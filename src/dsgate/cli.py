"""Command line: ``dsgate demo`` and ``dsgate readme --check|--write``."""

from __future__ import annotations

import sys
from pathlib import Path

BLOCKS = {"excerpt": ("<!-- demo-excerpt:start -->", "<!-- demo-excerpt:end -->", "README_excerpt.md"),
          "full": ("<!-- demo-output:start -->", "<!-- demo-output:end -->", "README_block.md")}


def _split(text: str, start: str, end: str) -> tuple[str, str, str]:
    head, rest = text.split(start, 1)
    current, tail = rest.split(end, 1)
    return head, current, tail


def _fenced(text: str) -> str:
    """The code blocks of a README section, without the provenance line above them."""
    parts = text.split("```text", 1)
    return ("```text" + parts[1]).strip() if len(parts) == 2 else ""


def readme(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="dsgate readme")
    ap.add_argument("--dir", default="build/demo", help="demo output holding README_excerpt.md and README_block.md")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--provenance", default="", help="line written above the excerpt (--write only)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    text = Path(args.readme).read_text()
    if args.check:
        bad = [name for name, (start, end, fname) in BLOCKS.items()
               if _fenced(_split(text, start, end)[1]) != (Path(args.dir) / fname).read_text().strip()]
        if bad:
            print(f"README blocks differ from this run's demo output: {', '.join(bad)}", file=sys.stderr)
            return 1
        print("README blocks match this run's demo output")
        return 0
    for name, (start, end, fname) in BLOCKS.items():
        head, _, tail = _split(text, start, end)
        prov = (args.provenance + "\n\n") if (args.provenance and name == "excerpt") else ""
        text = f"{head}{start}\n{prov}{(Path(args.dir) / fname).read_text()}{end}{tail}"
    Path(args.readme).write_text(text)
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
