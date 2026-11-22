"""MkDocs hook: render the ``neuroprep-agentic`` command reference from the real parser.

``cli()`` builds its argparse parser inline, so the hook intercepts ``parse_args`` to
capture it instead of duplicating the definitions; the page can never drift from the code.
"""

from __future__ import annotations

import argparse
import os

MARKER = "<!-- cli-reference -->"


class _Captured(Exception):
    def __init__(self, parser: argparse.ArgumentParser) -> None:
        self.parser = parser


def _capture_parser() -> argparse.ArgumentParser:
    from neuroprep_agentic.main import cli

    original = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        raise _Captured(self)

    argparse.ArgumentParser.parse_args = capture
    try:
        cli([])
    except _Captured as captured:
        return captured.parser
    finally:
        argparse.ArgumentParser.parse_args = original
    raise RuntimeError("neuroprep_agentic.main.cli did not parse arguments")


def _render() -> str:
    os.environ["COLUMNS"] = "88"
    parser = _capture_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    summaries = {choice.dest: choice.help or "" for choice in subparsers._choices_actions}
    lines = ["| Command | Purpose |", "|---|---|"]
    for name in subparsers.choices:
        summary = " ".join(summaries.get(name, "").split()).replace("|", "\\|")
        lines.append(f"| [`{name}`](#{name}) | {summary} |")
    lines.append("")
    for name, sub in subparsers.choices.items():
        lines += [f"## `{name}` {{#{name}}}", ""]
        if summaries.get(name):
            lines += [" ".join(summaries[name].split()).rstrip(".") + ".", ""]
        help_text = sub.format_help().replace("usage: ", "", 1).rstrip()
        lines += ["```text", help_text, "```", ""]
    return "\n".join(lines)


def on_page_markdown(markdown: str, page, config, files) -> str:
    if MARKER not in markdown:
        return markdown
    return markdown.replace(MARKER, _render())
