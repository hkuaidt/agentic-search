"""Execute every ```python block of docs/tutorial.md so the tutorial cannot drift from the code.

Blocks run in order within a chapter (``## `` heading) in one shared namespace, like a notebook.
Blocks whose first line contains "需要 API" need a real model endpoint and are skipped.
"""

import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parents[1] / "docs" / "tutorial.md"


def _chapters():
    text = DOC.read_text(encoding="utf-8")
    for part in re.split(r"^## ", text, flags=re.M)[1:]:
        title = part.splitlines()[0].strip()
        blocks = re.findall(r"```python\n(.*?)```", part, flags=re.S)
        blocks = [b for b in blocks if "需要 API" not in b.splitlines()[0]]
        if blocks:
            yield pytest.param(blocks, id=title)


@pytest.mark.parametrize("blocks", list(_chapters()))
def test_tutorial_chapter_runs(blocks):
    ns: dict = {"__name__": "tutorial"}
    for i, block in enumerate(blocks):
        exec(compile(block, f"tutorial-block-{i}", "exec"), ns)


def test_tutorial_covers_every_role():
    titles = [m.group(1) for m in re.finditer(r"^## (第 \d 章：.*)$", DOC.read_text(encoding="utf-8"), re.M)]
    for role in ("A 标签架构师", "B Agent 导演", "C 搜索管家", "D 裁判长", "E 优势分配官", "F 数据炼金师"):
        assert any(role in t for t in titles), role
