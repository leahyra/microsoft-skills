#!/usr/bin/env python3
"""
Synchronize .github/skills/<skill-name> links from plugin skill sources.

Examples:
  python .github/scripts/sync_skill_links.py --plugin azure-sdk-rust --check
  python .github/scripts/sync_skill_links.py --plugin azure-sdk-rust --apply
  python .github/scripts/sync_skill_links.py --all-plugins --check
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Drift:
    skill: str
    reason: str


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def plugin_skill_dirs(root: pathlib.Path, plugin: str) -> List[pathlib.Path]:
    skills_root = root / ".github" / "plugins" / plugin / "skills"
    if not skills_root.exists():
        return []
    result: List[pathlib.Path] = []
    for entry in sorted(skills_root.iterdir()):
        if entry.is_dir() and (entry / "SKILL.md").exists():
            result.append(entry)
    return result


def selected_plugins(
    root: pathlib.Path, plugin: str | None, all_plugins: bool
) -> List[str]:
    if plugin:
        return [plugin]
    if not all_plugins:
        raise ValueError("Provide --plugin <name> or --all-plugins")

    plugins_root = root / ".github" / "plugins"
    names: List[str] = []
    for entry in sorted(plugins_root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "skills").exists():
            names.append(entry.name)
    return names


def expected_link_target(link_path: pathlib.Path, src_skill_dir: pathlib.Path) -> str:
    return os.path.relpath(src_skill_dir, start=link_path.parent)


def readlink_target(path: pathlib.Path) -> str | None:
    if not path.exists() and not path.is_symlink():
        return None
    if not path.is_symlink():
        return None
    return os.readlink(path)


def ensure_symlink(link_path: pathlib.Path, source_dir: pathlib.Path) -> None:
    desired = expected_link_target(link_path, source_dir)

    if link_path.is_symlink():
        current = os.readlink(link_path)
        if current == desired:
            return
        link_path.unlink()
    elif link_path.exists():
        raise RuntimeError(
            f"Cannot create link at {link_path}: existing non-symlink path present"
        )

    os.symlink(desired, link_path, target_is_directory=True)


def check_plugin(root: pathlib.Path, plugin: str) -> List[Drift]:
    drifts: List[Drift] = []
    dest_root = root / ".github" / "skills"

    for src in plugin_skill_dirs(root, plugin):
        skill = src.name
        link_path = dest_root / skill
        desired = expected_link_target(link_path, src)

        if not link_path.exists() and not link_path.is_symlink():
            drifts.append(Drift(skill, "missing link"))
            continue

        if not link_path.is_symlink():
            drifts.append(Drift(skill, "path exists but is not a symlink"))
            continue

        current = os.readlink(link_path)
        if current != desired:
            drifts.append(Drift(skill, f"wrong target: {current} (expected {desired})"))

    return drifts


def apply_plugin(root: pathlib.Path, plugin: str) -> List[Drift]:
    drifts: List[Drift] = []
    dest_root = root / ".github" / "skills"
    dest_root.mkdir(parents=True, exist_ok=True)

    for src in plugin_skill_dirs(root, plugin):
        skill = src.name
        link_path = dest_root / skill
        try:
            ensure_symlink(link_path, src)
        except Exception as ex:
            drifts.append(Drift(skill, str(ex)))
    return drifts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync plugin skill links into .github/skills"
    )
    parser.add_argument("--plugin", help="Single plugin name (e.g., azure-sdk-rust)")
    parser.add_argument(
        "--all-plugins", action="store_true", help="Process all plugins with skills/"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check only; fail on drift"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Create/update missing links"
    )
    args = parser.parse_args()

    if args.check == args.apply:
        print("error: provide exactly one of --check or --apply", file=sys.stderr)
        return 2

    root = repo_root()

    try:
        plugins = selected_plugins(root, args.plugin, args.all_plugins)
    except ValueError as ex:
        print(f"error: {ex}", file=sys.stderr)
        return 2

    any_drift = False
    for plugin in plugins:
        if args.check:
            drifts = check_plugin(root, plugin)
            if drifts:
                any_drift = True
                print(f"[DRIFT] plugin={plugin}")
                for d in drifts:
                    print(f"  - {d.skill}: {d.reason}")
            else:
                print(f"[OK] plugin={plugin}")
        else:
            drifts = apply_plugin(root, plugin)
            if drifts:
                any_drift = True
                print(f"[ERROR] plugin={plugin}")
                for d in drifts:
                    print(f"  - {d.skill}: {d.reason}")
            else:
                print(f"[SYNCED] plugin={plugin}")

    return 1 if any_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
