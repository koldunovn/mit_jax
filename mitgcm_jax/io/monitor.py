"""Parse MITgcm monitor output (`%MON name = value` lines in STDOUT.0000) and cg2d solver lines.

`read_monitor(path)` returns {iteration: {name: float}}, keyed by `time_tsnumber`; each `%MON` block starts with
`time_tsnumber`. Also collects per-step `cg2d_init_res`, `cg2d_iters` and `cg2d_res` lines (printed in the monitor
block of the step) under the same keys. Values keep full printed precision (14 digits in c66g).
"""

import re
from pathlib import Path

_MON = re.compile(r"%MON\s+(\w+)\s*=\s*(\S+)")
_CG2D = re.compile(r"\s(cg2d_init_res|cg2d_iters|cg2d_res)\s*=\s*(\S+)")


def _num(s):
    try:
        return float(s.replace("D", "E"))
    except ValueError:
        return float("nan")


def read_monitor(path, prefix="%MON"):
    out, cur = {}, None
    for line in Path(path).read_text(errors="replace").splitlines():
        m = _MON.search(line)
        if m:
            name, val = m.group(1), _num(m.group(2))
            if name == "time_tsnumber":
                cur = out.setdefault(int(val), {})
            if cur is not None:
                cur[name] = val
            continue
        m = _CG2D.search(line)
        if m and cur is not None and "=" in line and "%" not in line:
            cur[m.group(1)] = _num(m.group(2))
    return out


def compare_monitors(a, b):
    """Per (iteration, name) present in both: relative difference |x-y| / max(|x|, |y|, tiny). Returns
    {(it, name): rel} and the list of names present in only one of the two."""
    diffs, only = {}, set()
    for it in sorted(set(a) & set(b)):
        for name in set(a[it]) | set(b[it]):
            if name not in a[it] or name not in b[it]:
                only.add(name)
                continue
            x, y = a[it][name], b[it][name]
            den = max(abs(x), abs(y), 1e-300)
            diffs[(it, name)] = abs(x - y) / den
    return diffs, sorted(only)
