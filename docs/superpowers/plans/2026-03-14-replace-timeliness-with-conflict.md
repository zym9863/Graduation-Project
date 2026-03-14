# Replace timeliness with conflict — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `timeliness` news value dimension with `conflict`, update the `proximity` heuristic to use livelihood keywords, and enhance the LLM system prompt with detailed scoring criteria for all five dimensions.

**Architecture:** Swap dimension names and heuristic logic in the annotator module, update the JSON parse function's field mapping, replace the system prompt with an enhanced version containing per-dimension rubrics, and propagate the rename through tests and documentation.

**Tech Stack:** Python 3.12, pytest, OpenAI-compatible API client

---

## File Map

| File | Role | Action |
|------|------|--------|
| `src/features/news_value_annotator.py` | Core annotator: prompt, heuristic, parser | Modify |
| `scripts/annotate_news_value.py` | CLI script: dimension names tuple | Modify |
| `tests/test_news_value_annotator.py` | Unit tests for parser and heuristic | Modify |
| `tests/test_annotate_news_value.py` | Integration tests for CLI | Modify |
| `README.md` | Chinese documentation | Modify |
| `README-EN.md` | English documentation | Modify |

---

## Chunk 1: Core annotator + unit tests

### Task 1: Update parse function tests (TDD — write failing tests first)

**Files:**
- Modify: `tests/test_news_value_annotator.py`

- [ ] **Step 1: Rewrite test file with new dimension names**

Replace the entire test file contents with:

```python
import pytest

from src.features.news_value_annotator import parse_news_value_response


def test_parse_news_value_response_with_english_fields() -> None:
    raw = '{"conflict": 5, "importance": 4, "prominence": 3, "proximity": 2, "interest": 1}'

    scores = parse_news_value_response(raw)

    assert scores == [5, 4, 3, 2, 1]


def test_parse_news_value_response_with_chinese_fields() -> None:
    raw = '{"冲突性": 4, "重要性": 4, "显著性": 3, "接近性": 2, "趣味性": 5}'

    scores = parse_news_value_response(raw)

    assert scores == [4, 4, 3, 2, 5]


def test_parse_news_value_response_clamps_to_valid_range() -> None:
    raw = "[0, 2, 9, 5, 7]"

    scores = parse_news_value_response(raw)

    assert scores == [1, 2, 5, 5, 5]


def test_parse_news_value_response_rejects_missing_fields() -> None:
    raw = '{"conflict": 5, "importance": 4, "prominence": 3}'

    with pytest.raises(ValueError):
        parse_news_value_response(raw)
```

- [ ] **Step 1b: Add heuristic unit tests (will also fail initially)**

Append the following tests to the same file:

```python
from src.features.news_value_annotator import heuristic_news_value_scores


def test_heuristic_conflict_default_is_two() -> None:
    article = {"title": "Company releases quarterly earnings", "abstract": "Steady growth reported.", "category": "finance"}

    scores = heuristic_news_value_scores(article)

    assert scores[0] == 2  # conflict: no conflict keywords → default 2


def test_heuristic_conflict_keyword_scores_four() -> None:
    article = {"title": "War erupts in border region", "abstract": "Military conflict escalates.", "category": "news"}

    scores = heuristic_news_value_scores(article)

    assert scores[0] == 4  # conflict: "war" and "conflict" present → 4


def test_heuristic_proximity_livelihood_keyword_scores_four() -> None:
    article = {"title": "Housing prices surge nationwide", "abstract": "Rent increases affect millions.", "category": "finance"}

    scores = heuristic_news_value_scores(article)

    assert scores[3] == 4  # proximity: "housing" and "rent" present → 4


def test_heuristic_proximity_default_is_three() -> None:
    article = {"title": "New art exhibition opens", "abstract": "Local gallery showcases paintings.", "category": "lifestyle"}

    scores = heuristic_news_value_scores(article)

    assert scores[3] == 3  # proximity: no livelihood keywords → default 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_news_value_annotator.py -v`
Expected: `test_parse_news_value_response_with_english_fields` and `test_parse_news_value_response_with_chinese_fields` FAIL (old parser looks for `timeliness`/`时新性` keys). All four `test_heuristic_*` tests FAIL (old function uses `timeliness` variable, default differs, and `proximity` is hardcoded to 3).

### Task 2: Update parse function and prompt in annotator

**Files:**
- Modify: `src/features/news_value_annotator.py` (SYSTEM_PROMPT constant)
- Modify: `src/features/news_value_annotator.py` (parse field mapping line)

- [ ] **Step 3: Replace SYSTEM_PROMPT**

In `src/features/news_value_annotator.py`, find and replace the `SYSTEM_PROMPT` string:

```python
SYSTEM_PROMPT = """你是新闻价值标注器。请基于给定新闻内容，对以下五个要素分别打 1 到 5 分整数：timeliness、importance、prominence、proximity、interest。只返回 JSON。"""
```

With:

```python
SYSTEM_PROMPT = """\
你是新闻价值标注器。请基于给定新闻的标题、摘要和类别信息，对以下五个维度分别打 1-5 分整数。

## 评分维度

1. conflict（冲突性）：新闻是否涉及对抗、争端、分歧或紧张关系。
   - 1 分：无任何冲突元素
   - 3 分：存在温和分歧或潜在矛盾
   - 5 分：涉及激烈对抗、战争、诉讼等严重冲突

2. importance（重要性）：事件对社会、经济或公众生活的影响程度。
   - 1 分：影响极小，仅涉及个别人
   - 3 分：影响一定范围的群体或行业
   - 5 分：影响国家或全球范围的重大事件

3. prominence（显著性）：涉及人物、机构或地点的知名度。
   - 1 分：涉及普通个人或无名机构
   - 3 分：涉及地区知名人物或机构
   - 5 分：涉及国际级知名人物、领导人或机构

4. proximity（接近性）：事件与普通大众日常生活的心理关联程度。
   - 1 分：与大多数人生活无关
   - 3 分：与部分人群的日常生活相关
   - 5 分：直接关系到每个人的切身利益

5. interest（趣味性）：内容的吸引力和可读性。
   - 1 分：枯燥乏味，无吸引力
   - 3 分：有一定趣味或话题性
   - 5 分：极具吸引力，引人入胜

## 输出格式

只返回 JSON 对象，格式如下：
{"conflict": <int>, "importance": <int>, "prominence": <int>, "proximity": <int>, "interest": <int>}"""
```

- [ ] **Step 4: Update parse_news_value_response field mapping**

In `src/features/news_value_annotator.py`, find the line inside `parse_news_value_response`:

```python
            parsed.get("timeliness") or parsed.get("时新性"),
```

Replace with:

```python
            parsed.get("conflict") or parsed.get("冲突性"),
```

- [ ] **Step 5: Run parser tests to verify they pass**

Run: `uv run pytest tests/test_news_value_annotator.py::test_parse_news_value_response_with_english_fields tests/test_news_value_annotator.py::test_parse_news_value_response_with_chinese_fields tests/test_news_value_annotator.py::test_parse_news_value_response_clamps_to_valid_range tests/test_news_value_annotator.py::test_parse_news_value_response_rejects_missing_fields -v`
Expected: All 4 parser tests PASS. The heuristic tests still fail (not yet implemented).

- [ ] **Step 6: Commit**

```bash
git add src/features/news_value_annotator.py tests/test_news_value_annotator.py
git commit -m "feat: replace timeliness with conflict in parser and prompt"
```

### Task 3: Update heuristic function (conflict + proximity)

**Files:**
- Modify: `src/features/news_value_annotator.py` (heuristic function and module-level keyword constants)

- [ ] **Step 7: Replace heuristic function body**

In `src/features/news_value_annotator.py`, find the entire `heuristic_news_value_scores` function (it will be below the expanded SYSTEM_PROMPT):

```python
def heuristic_news_value_scores(article: dict[str, str]) -> list[int]:
    text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
    timeliness = 4 if any(word in text for word in ("breaking", "today", "latest", "new", "just")) else 3
    importance = 4 if article.get("category") in {"news", "finance", "health"} else 3
    prominence = 4 if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", article.get("title", "")) else 3
    proximity = 3
    interest = 4 if article.get("category") in {"sports", "entertainment", "lifestyle"} else 3
    return [timeliness, importance, prominence, proximity, interest]
```

With:

```python
_CONFLICT_KEYWORDS = frozenset((
    "war", "conflict", "dispute", "protest", "lawsuit", "attack", "oppose",
    "fight", "debate", "crisis", "strike", "clash", "battle", "sue",
    "accuse", "condemn", "threat", "sanction", "ban", "arrest",
))

_PROXIMITY_KEYWORDS = frozenset((
    "price", "tax", "job", "employment", "wage", "salary", "health",
    "hospital", "school", "education", "housing", "rent", "food",
    "inflation", "insurance", "retirement", "pension", "traffic", "commute",
))


def heuristic_news_value_scores(article: dict[str, str]) -> list[int]:
    text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
    conflict = 4 if any(word in text for word in _CONFLICT_KEYWORDS) else 2
    importance = 4 if article.get("category") in {"news", "finance", "health"} else 3
    prominence = 4 if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", article.get("title", "")) else 3
    proximity = 4 if any(word in text for word in _PROXIMITY_KEYWORDS) else 3
    interest = 4 if article.get("category") in {"sports", "entertainment", "lifestyle"} else 3
    return [conflict, importance, prominence, proximity, interest]
```

Note: The keyword frozensets are defined at module level (between `SYSTEM_PROMPT` and the function) to avoid re-creating them on every call.

- [ ] **Step 8: Run ALL annotator tests (parser + heuristic)**

Run: `uv run pytest tests/test_news_value_annotator.py -v`
Expected: All 8 tests PASS (4 parser + 4 heuristic).

- [ ] **Step 9: Commit**

```bash
git add src/features/news_value_annotator.py
git commit -m "feat: update heuristic with conflict keywords and proximity livelihood keywords"
```

---

## Chunk 2: CLI script + integration tests

### Task 4: Update integration tests first (TDD — write failing tests)

**Files:**
- Modify: `tests/test_annotate_news_value.py`

- [ ] **Step 10: Rewrite integration test file with new dimension assertions**

Replace the entire test file contents with:

```python
import pytest

from scripts.annotate_news_value import main, parse_args


def test_parse_args_single_mode_enabled() -> None:
    args = parse_args([
        "--single-title",
        "政策发布",
        "--single-abstract",
        "影响多个行业",
    ])

    assert args.single_mode is True


def test_parse_args_single_mode_conflict_with_limit() -> None:
    with pytest.raises(SystemExit):
        parse_args([
            "--single-title",
            "政策发布",
            "--limit",
            "1",
        ])


def test_single_case_prints_named_vector(capsys: pytest.CaptureFixture[str]) -> None:
    main([
        "--provider",
        "heuristic",
        "--single-title",
        "Breaking: New regulation released",
        "--single-abstract",
        "Authorities announced a major update today.",
        "--single-category",
        "news",
    ])

    output = capsys.readouterr().out
    assert "Single News Value Case" in output
    assert '"conflict"' in output
    assert '"importance"' in output
    assert '"prominence"' in output
    assert '"proximity"' in output
    assert '"interest"' in output
    assert "vector:" in output


def test_single_case_conflict_keyword_scores_high(capsys: pytest.CaptureFixture[str]) -> None:
    main([
        "--provider",
        "heuristic",
        "--single-title",
        "War breaks out between two nations",
        "--single-abstract",
        "A major military conflict erupted today.",
        "--single-category",
        "news",
    ])

    output = capsys.readouterr().out
    assert '"conflict": 4' in output
```

- [ ] **Step 11: Run integration tests to verify they fail**

Run: `uv run pytest tests/test_annotate_news_value.py::test_single_case_prints_named_vector -v`
Expected: FAILS because CLI still outputs `"timeliness"` instead of `"conflict"`.

### Task 5: Update CLI dimension tuple

**Files:**
- Modify: `scripts/annotate_news_value.py`

- [ ] **Step 12: Replace VALUE_DIMENSIONS**

In `scripts/annotate_news_value.py`, find:

```python
VALUE_DIMENSIONS = ("timeliness", "importance", "prominence", "proximity", "interest")
```

Replace with:

```python
VALUE_DIMENSIONS = ("conflict", "importance", "prominence", "proximity", "interest")
```

- [ ] **Step 13: Run all tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 14: Commit**

```bash
git add scripts/annotate_news_value.py tests/test_annotate_news_value.py
git commit -m "feat: update CLI dimensions tuple and integration tests for conflict"
```

---

## Chunk 3: Documentation

### Task 6: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 15: Update Chinese README dimension list**

In `README.md`, find:

```markdown
- 五维价值打分（`timeliness`、`importance`、`prominence`、`proximity`、`interest`）
```

Replace with:

```markdown
- 五维价值打分（`conflict`、`importance`、`prominence`、`proximity`、`interest`）
```

### Task 7: Update README-EN.md

**Files:**
- Modify: `README-EN.md`

- [ ] **Step 16: Update English README dimension list**

In `README-EN.md`, find:

```markdown
- Five-dimensional value scores (`timeliness`, `importance`, `prominence`, `proximity`, `interest`)
```

Replace with:

```markdown
- Five-dimensional value scores (`conflict`, `importance`, `prominence`, `proximity`, `interest`)
```

- [ ] **Step 17: Run full test suite to confirm nothing is broken**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 18: Commit**

```bash
git add README.md README-EN.md
git commit -m "docs: update dimension names from timeliness to conflict in READMEs"
```

---

## Post-Implementation Note

After all code changes are committed, the existing `data/news_value_scores.json` file (if present) must be **regenerated**. The old data has timeliness scores in the first position, which is now semantically invalid. Run:

```bash
uv run python main.py annotate-news-value --provider heuristic --overwrite
```

This is an operational step, not a code change.
