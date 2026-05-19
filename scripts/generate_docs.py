#!/usr/bin/env python3
"""Generate per-command MDX reference pages for the website.

Walks the Click command tree exposed by :mod:`cents.cli` and emits one MDX
file per top-level command (or group) under
``website/src/content/docs/commands/``.

Hand-written prose can live next to the generated file as
``{name}.intro.mdx``. When present, its content is prepended to the body of
the generated file so regeneration never clobbers human-written narrative.

Run from a fresh checkout (no install required)::

    python scripts/generate_docs.py [output_dir]

The generator is idempotent: running it twice produces byte-identical files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

# Make the in-tree `src/cents` importable without requiring `pip install -e .`
# so contributors can regenerate docs straight from a clone.
_REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

import click

from cents.cli import cli


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _escape_mdx_table_cell(text: str) -> str:
    """Escape a string so it survives a markdown table cell."""
    if text is None:
        return ""
    # Escape pipe; collapse newlines to spaces; escape angle brackets that
    # MDX would otherwise try to interpret as JSX.
    cleaned = text.replace("\n", " ").replace("|", "\\|")
    return cleaned.replace("<", "&lt;").replace(">", "&gt;")


_UNSET_SENTINEL = getattr(click.core, "UNSET", None)


def _format_default(option: click.Option) -> str:
    """Render an option's default value for documentation."""
    default = option.default
    if default is None or default is _UNSET_SENTINEL:
        return ""
    if option.is_flag or isinstance(default, bool):
        return "true" if default else "false"
    if callable(default):
        return ""
    return str(default)


def _format_type(param: click.Parameter) -> str:
    """Render a parameter's type for documentation."""
    t = param.type
    if isinstance(t, click.Choice):
        return "[" + " \\| ".join(t.choices) + "]"
    if isinstance(t, click.Tuple):
        inner = " ".join(_inner_type_name(child) for child in t.types)
        return f"({inner})"
    return _inner_type_name(t)


def _inner_type_name(t: click.ParamType) -> str:
    name = getattr(t, "name", None) or t.__class__.__name__.lower()
    return name


_DUMMY_CTX = click.Context(click.Command("__doc__"))


def _option_signature(option: click.Option) -> str:
    """Render an option's invocation form (e.g., ``--foo BAR``)."""
    flags = "/".join(sorted(option.opts, key=len, reverse=True))
    if option.is_flag:
        return flags
    metavar = option.make_metavar(ctx=_DUMMY_CTX)
    return f"{flags} {metavar}".strip()


def _argument_signature(arg: click.Argument) -> str:
    name = (arg.name or "").upper()
    if arg.nargs == -1:
        return f"[{name}...]"
    if arg.required:
        return name
    return f"[{name}]"


def _command_synopsis(prog: str, cmd: click.Command) -> str:
    """Render a usage string for a command."""
    parts = [prog]
    args = [p for p in cmd.params if isinstance(p, click.Argument)]
    options = [p for p in cmd.params if isinstance(p, click.Option)]
    if options:
        parts.append("[OPTIONS]")
    parts.extend(_argument_signature(a) for a in args)
    return " ".join(parts)


def _options_table(cmd: click.Command) -> str:
    """Render a command's options as a markdown table."""
    options = [p for p in cmd.params if isinstance(p, click.Option)]
    if not options:
        return ""
    rows = [
        "| Option | Type | Default | Description |",
        "| --- | --- | --- | --- |",
    ]
    for opt in options:
        signature = "`" + _option_signature(opt) + "`"
        type_str = _format_type(opt)
        type_cell = f"`{type_str}`" if type_str else ""
        default_str = _format_default(opt)
        default_cell = f"`{default_str}`" if default_str else ""
        help_text = _escape_mdx_table_cell(opt.help or "")
        rows.append(f"| {signature} | {type_cell} | {default_cell} | {help_text} |")
    return "\n".join(rows)


def _arguments_table(cmd: click.Command) -> str:
    """Render a command's positional arguments as a markdown table."""
    args = [p for p in cmd.params if isinstance(p, click.Argument)]
    if not args:
        return ""
    rows = [
        "| Argument | Type | Required |",
        "| --- | --- | --- |",
    ]
    for arg in args:
        name = "`" + (arg.name or "").upper() + "`"
        type_str = _format_type(arg)
        type_cell = f"`{type_str}`" if type_str else ""
        required = "yes" if arg.required else "no"
        rows.append(f"| {name} | {type_cell} | {required} |")
    return "\n".join(rows)


def _example_for(cmd_path: str, cmd: click.Command) -> str:
    """Construct a sensible usage example for a command.

    Prefer an example pulled from the command's help text when one is
    embedded (heuristic: a line beginning with ``Example:``). Otherwise
    construct a synthesized one from the synopsis.
    """
    help_text = cmd.help or ""
    for line in help_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("example:"):
            return stripped.split(":", 1)[1].strip()
    return _command_synopsis(f"cents {cmd_path}", cmd)


def _short_description(cmd: click.Command) -> str:
    """Return the first non-empty line of a command's help text."""
    help_text = (cmd.help or cmd.short_help or "").strip()
    if not help_text:
        return ""
    for line in help_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _frontmatter(title: str, description: str) -> str:
    """Render YAML frontmatter for a Starlight MDX page."""
    # Quote and escape any embedded double quotes
    safe_title = title.replace('"', '\\"')
    safe_desc = description.replace('"', '\\"').replace("\n", " ")
    return f"""---
title: "{safe_title}"
description: "{safe_desc}"
---
"""


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------


def _render_command_section(parent_path: str, cmd: click.Command) -> str:
    """Render a command (or sub-command) as a MDX section."""
    name = cmd.name or ""
    full = f"{parent_path} {name}".strip() if parent_path else name
    help_text = (cmd.help or "").strip()
    sections: list[str] = []
    sections.append(f"### `cents {full}`")
    if help_text:
        sections.append(help_text)
    sections.append("**Synopsis**")
    sections.append("```bash\n" + _command_synopsis(f"cents {full}", cmd) + "\n```")
    args = _arguments_table(cmd)
    if args:
        sections.append("**Arguments**\n\n" + args)
    options = _options_table(cmd)
    if options:
        sections.append("**Options**\n\n" + options)
    sections.append("**Example**")
    sections.append("```bash\n" + _example_for(full, cmd) + "\n```")
    return "\n\n".join(sections)


def _render_group_body(group_name: str, group: click.Group) -> str:
    """Render the body for a Click Group page."""
    sections: list[str] = []
    help_text = (group.help or "").strip()
    if help_text:
        sections.append(help_text)
    sections.append("## Synopsis")
    sections.append(
        "```bash\ncents " + group_name + " <subcommand> [OPTIONS] [ARGS]...\n```"
    )
    if group.params:
        opts = _options_table(group)
        if opts:
            sections.append("## Group options\n\n" + opts)
    sub_names = sorted(group.commands.keys())
    if sub_names:
        sections.append("## Subcommands")
        bullets = []
        for sub_name in sub_names:
            sub = group.commands[sub_name]
            short = _short_description(sub)
            if short:
                bullets.append(f"- `cents {group_name} {sub_name}` — {short}")
            else:
                bullets.append(f"- `cents {group_name} {sub_name}`")
        sections.append("\n".join(bullets))
        for sub_name in sub_names:
            sub = group.commands[sub_name]
            sections.append(_render_command_section(group_name, sub))
    return "\n\n".join(sections)


def _render_command_body(cmd_name: str, cmd: click.Command) -> str:
    """Render the body for a non-group Click command page."""
    sections: list[str] = []
    help_text = (cmd.help or "").strip()
    if help_text:
        sections.append(help_text)
    sections.append("## Synopsis")
    sections.append("```bash\n" + _command_synopsis(f"cents {cmd_name}", cmd) + "\n```")
    args = _arguments_table(cmd)
    if args:
        sections.append("## Arguments\n\n" + args)
    opts = _options_table(cmd)
    if opts:
        sections.append("## Options\n\n" + opts)
    sections.append("## Example")
    sections.append("```bash\n" + _example_for(cmd_name, cmd) + "\n```")
    return "\n\n".join(sections)


def _render_page(name: str, cmd: click.Command, intro: str | None) -> str:
    """Render a complete MDX page for a top-level command/group."""
    title = f"cents {name}"
    description = _short_description(cmd) or f"Reference for `cents {name}`."
    frontmatter = _frontmatter(title, description)
    if isinstance(cmd, click.Group):
        body = _render_group_body(name, cmd)
    else:
        body = _render_command_body(name, cmd)
    parts: list[str] = [frontmatter]
    if intro:
        parts.append(intro.strip())
    parts.append(body)
    # Single trailing newline keeps the file POSIX-friendly and idempotent.
    return "\n\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# File system orchestration
# ---------------------------------------------------------------------------


def _iter_top_level_commands(root: click.Group) -> Iterable[tuple[str, click.Command]]:
    for name in sorted(root.commands.keys()):
        yield name, root.commands[name]


def _read_intro(commands_dir: Path, name: str) -> str | None:
    """Read a sibling ``_{name}.intro.mdx`` if present.

    The underscore prefix keeps the file out of Astro's content collection
    glob (Starlight ignores ``_``-prefixed entries) so the intro source
    doesn't try to render as its own doc page.
    """
    try:
        return (commands_dir / f"_{name}.intro.mdx").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def generate(output_root: Path, *, root: click.Group | None = None) -> list[Path]:
    """Generate one MDX file per top-level command.

    Args:
        output_root: Root directory under which ``src/content/docs/commands``
            will be created (typically the ``website`` directory).
        root: Click group to walk. Defaults to the top-level ``cents`` CLI.

    Returns:
        List of paths written, in deterministic order.
    """
    if root is None:
        root = cli
    commands_dir = output_root / "src" / "content" / "docs" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, cmd in _iter_top_level_commands(root):
        intro = _read_intro(commands_dir, name)
        content = _render_page(name, cmd, intro)
        target = commands_dir / f"{name}.mdx"
        # Skip the write when content is byte-identical so downstream caches
        # (mtime-keyed build pipelines) don't churn unnecessarily.
        try:
            existing = target.read_text(encoding="utf-8")
        except FileNotFoundError:
            existing = None
        if existing != content:
            target.write_text(content, encoding="utf-8")
        written.append(target)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="website",
        help="Output root directory (default: website)",
    )
    args = parser.parse_args(argv)
    output_root = Path(args.output_dir).resolve()
    written = generate(output_root)
    print(f"Wrote {len(written)} command reference pages to {output_root}")
    for path in written:
        print(f"  - {path.relative_to(output_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
