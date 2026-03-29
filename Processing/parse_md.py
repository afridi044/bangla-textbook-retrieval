#!/usr/bin/env python3
"""Parse Markdown headings using CommonMark and output a nested tree with content + optional JSON.

Usage:
  python parse_md.py path/to/file.md [--json-out out.json]

"""
import argparse
import json
import re
from pathlib import Path


def collect_sections(md_text):
    """Collect sections with headings and their content."""
    heading_pattern = r'^(#{1,6})\s+(.+)$'
    lines = md_text.split('\n')
    sections = []
    current_content = []
    current_level = 0
    current_heading = None
    for line in lines:
        match = re.match(heading_pattern, line)
        if match:
            # Save previous section if any
            if current_heading is not None:
                content = '\n'.join(current_content).strip()
                sections.append({'level': current_level, 'heading': current_heading, 'content': content})
            # Start new section
            level = len(match.group(1))
            heading = match.group(2).strip()
            current_level = level
            current_heading = heading
            current_content = []
        else:
            current_content.append(line)
    # Last section
    if current_heading is not None:
        content = '\n'.join(current_content).strip()
        sections.append({'level': current_level, 'heading': current_heading, 'content': content})
    return sections


def build_tree(sections):
    root = {'level': 0, 'heading': None, 'content': None, 'children': []}
    stack = [root]
    for s in sections:
        node = {'level': s['level'], 'heading': s['heading'], 'content': s['content'], 'children': []}
        # pop until top has smaller level
        while stack and stack[-1]['level'] >= node['level']:
            stack.pop()
        stack[-1]['children'].append(node)
        stack.append(node)
    return root


def print_tree(node, indent=0):
    if node['heading'] is not None:
        print('  ' * indent + f"- {node['heading']}")
        indent += 1
    for child in node['children']:
        print_tree(child, indent)


def main():
    ap = argparse.ArgumentParser(description='Parse Markdown headings (CommonMark)')
    ap.add_argument('mdfile', help='Path to Markdown file')
    ap.add_argument('--json-out', help='Write JSON output to this file', default=None)
    args = ap.parse_args()

    p = Path(args.mdfile)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")

    text = p.read_text(encoding='utf-8')
    sections = collect_sections(text)
    tree = build_tree(sections)

    json_out = json.dumps(tree['children'], ensure_ascii=False, indent=2)

    # If a JSON output path is provided, write the file and remain silent.
    if args.json_out:
        Path(args.json_out).write_text(json_out, encoding='utf-8')
        # exit without printing the tree/JSON to stdout
        return

    # default behavior: print human-friendly tree and JSON
    print('Heading tree:')
    print_tree(tree)
    print()
    print('JSON (top-level headings):')
    print(json_out)


if __name__ == '__main__':
    main()
