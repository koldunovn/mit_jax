"""docs/OVERRIDES.md must describe exactly the current override trees (plan Task 2).

The real check fails if a changed override file has no classification row, a row names an unchanged file, or a
generated table is stale. The negative controls rebuild the same failures on a synthetic c66g + override tree.
"""

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("audit_overrides", REPO / "scripts" / "audit_overrides.py")
ao = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ao)


def test_overrides_doc_is_complete_and_current():
    for path in (ao.C66G, *ao.TREES.values()):
        assert path.is_dir(), f"source clone missing: {path} (see CATALOG.md section 1)"
    rows, crows = ao.inventory(), ao.cpp_rows()
    assert len(rows) > 70 and len(ao.changed_files(rows)) > 70  # guard against an empty scan passing
    problems = ao.check((REPO / "docs" / "OVERRIDES.md").read_text(), rows, crows)
    assert problems == []


def _fake_tree(tmp_path):
    c66g = tmp_path / "c66g"
    (c66g / "model/src").mkdir(parents=True)
    (c66g / "pkg/exf").mkdir(parents=True)
    (c66g / "model/src/step.F").write_text("C comment\n      x = 1\n")
    (c66g / "model/src/same.F").write_text("      y = 2\n")
    (c66g / "pkg/exf/EXF_OPTIONS.h").write_text("#define A\n#undef B\n#ifdef A\n#define C\n#else\n#define D\n#endif\n")
    full, ff = tmp_path / "full", tmp_path / "ff"
    for t in (full, ff):
        t.mkdir()
        (t / "same.F").write_text("      y = 2\n")
    (full / "step.F").write_text("C changed comment\n      x = 1\n")  # comment-only change
    (ff / "step.F").write_text("C comment\n      x = 10\n")         # code change
    (ff / "EXF_OPTIONS.h").write_text("#define A\n#define B\n#ifdef A\n#define C\n#else\n#define D\n#endif\n"
                                      "C#define FORTRAN_COMMENT\n")
    return c66g, {"full": full, "ff": ff}


def _doc(rows, crows, classified):
    body = "\n".join(f"| `{n}` | F X | fwd | x |" for n in classified)
    return (f"# t\n\n## Classification\n\n| file | trees | class | what |\n|---|---|---|---|\n{body}\n\n"
            f"## Inventory\n\n{ao.render(rows)}\n\n## CPP\n\n{ao.render_cpp(crows)}\n")


def test_negative_controls(tmp_path):
    c66g, trees = _fake_tree(tmp_path)
    rows, crows = ao.inventory(trees, c66g), ao.cpp_rows(trees, c66g)
    by_name = {r["name"]: r for r in rows}
    assert by_name["same.F"]["full"] == (0, 0) and by_name["same.F"]["ff"] == (0, 0)
    assert by_name["step.F"]["full"] == (2, 0)            # raw 2, code 0: comment-only
    assert by_name["step.F"]["ff"] == (2, 2)
    assert set(ao.changed_files(rows)) == {"step.F", "EXF_OPTIONS.h"}
    assert [(h, m) for h, m, *_ in crows] == [("EXF_OPTIONS.h", "B")]  # A, C, D equal; comment ignored

    good = _doc(rows, crows, ["step.F", "EXF_OPTIONS.h"])
    assert ao.check(good, rows, crows) == []
    missing_row = _doc(rows, crows, ["EXF_OPTIONS.h"])
    assert any("without a classification row" in p for p in ao.check(missing_row, rows, crows))
    extra_row = _doc(rows, crows, ["step.F", "EXF_OPTIONS.h", "same.F"])
    assert any("not changed" in p for p in ao.check(extra_row, rows, crows))
    stale = good.replace("| 2 (2) |", "| 1 (2) |")
    assert stale != good and any("inventory table is stale" in p for p in ao.check(stale, rows, crows))
    assert any("markers missing" in p for p in ao.check("## Classification\n", rows, crows))

    # a newly planted override file is caught
    (trees["ff"] / "new_routine.F").write_text("      z = 3\n")
    rows2, crows2 = ao.inventory(trees, c66g), ao.cpp_rows(trees, c66g)
    problems = ao.check(good, rows2, crows2)
    assert any("new_routine.F" in p for p in problems)
    assert any("inventory table is stale" in p for p in problems)


def test_cpp_macro_context():
    m = ao.cpp_macros("#ifndef X_H\n#ifdef ALLOW_EXF\n#ifdef P\n#define Q\n#else\n#undef Q\n#endif\n#endif\n#endif\n")
    assert m["Q"] == [("define", "!X_H & ALLOW_EXF & P"), ("undef", "!X_H & ALLOW_EXF & !(P)")]
    assert ao._state(m["Q"]) == "define [if P]; undef [if !(P)]"


def test_ambiguous_base_is_an_error(tmp_path):
    for sub in ("pkg/a", "pkg/b"):
        (tmp_path / sub).mkdir(parents=True)
        (tmp_path / sub / "dup.F").write_text("\n")
    with pytest.raises(RuntimeError, match="ambiguous"):
        ao.find_base("dup.F", tmp_path)
