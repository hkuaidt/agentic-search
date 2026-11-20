"""Three-level tag tree (L1 50+ / L2 700+ / L3 5000+ in the PROPOSAL setup).

Only L1 and L3 taggers are trained; L2 is kept as a structural buffer and recovered by lookup because
L3 -> L2 is a strict tree mapping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TagNode:
    name: str
    level: int  # 1, 2 or 3
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    description: str = ""
    keywords: list[str] = field(default_factory=list)


BLANK_TAG = "空白标签"  # emitted by the L3 tagger when no candidate fits (open-set discovery)


class TagTree:
    def __init__(self) -> None:
        self.nodes: dict[str, TagNode] = {}

    # -- construction -------------------------------------------------------------------------
    def add(self, name: str, level: int, parent: str | None = None, **kw) -> TagNode:
        if name in self.nodes:
            raise ValueError(f"duplicate tag name {name!r}; tag names must be globally unique")
        if level not in (1, 2, 3):
            raise ValueError("level must be 1, 2 or 3")
        if level > 1 and (
            parent is None or parent not in self.nodes or self.nodes[parent].level != level - 1
        ):
            raise ValueError(f"{name!r}: parent must be an existing level-{level - 1} tag")
        node = TagNode(name=name, level=level, parent=parent, **kw)
        self.nodes[name] = node
        if parent:
            self.nodes[parent].children.append(name)
        return node

    @classmethod
    def from_paths(cls, paths: list[tuple[str, str, str]]) -> TagTree:
        tree = cls()
        for l1, l2, l3 in paths:
            if l1 not in tree.nodes:
                tree.add(l1, 1)
            if l2 not in tree.nodes:
                tree.add(l2, 2, l1)
            tree.add(l3, 3, l2)
        return tree

    # -- queries ------------------------------------------------------------------------------
    def level(self, level: int) -> list[str]:
        return [n.name for n in self.nodes.values() if n.level == level]

    def l3_under_l1(self, l1: str) -> list[str]:
        """Candidate set for the L3 tagger once L1 is fixed (~90-100 in the PROPOSAL setup)."""
        return [l3 for l2 in self.nodes[l1].children for l3 in self.nodes[l2].children]

    def path(self, l3: str) -> tuple[str, str, str]:
        n3 = self.nodes[l3]
        if n3.level != 3 or n3.parent is None:
            raise ValueError(f"{l3!r} is not an L3 tag")
        n2 = self.nodes[n3.parent]
        assert n2.parent is not None
        return (n2.parent, n2.name, n3.name)

    def stats(self) -> dict[str, int]:
        return {f"L{k}": len(self.level(k)) for k in (1, 2, 3)}

    # -- io -----------------------------------------------------------------------------------
    def to_json(self) -> dict:
        return {"paths": [list(self.path(l3)) for l3 in self.level(3)]}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TagTree:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_paths([tuple(p) for p in data["paths"]])
