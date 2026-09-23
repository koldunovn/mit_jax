"""Reader for MITgcm run-time namelist files (`data`, `data.*`, `eedata`).

MITgcm pre-processes these files before the Fortran namelist read (`eesupp/src/nml_filter.F` /
`model/src/ini_parms.F`): lines whose first non-blank character is `#` are comments. What remains is standard
Fortran namelist syntax, used loosely: trailing commas are optional (entries may be separated by newlines only),
arrays may be written with repeat counts (`3*23.`), and keys may carry indices or index ranges
(`xx_genarr3d_file(3)`, `xx_genarr2d_bounds(1:5,1)`).

`read_namelist(path)` returns `{group: {key: [values]}}` with group and key names lower-cased and indices kept as
written (spaces removed), e.g. `{'parm01': {'tref': [23.0, 23.0, ...]}, 'ctrl_nml_genarr':
{'xx_genarr3d_file(1)': ['xx_theta']}}`. Values: `str` (quotes removed), `bool`, `int`, `float`.
A key assigned twice keeps the last assignment, as the Fortran read does.
"""

import re
from pathlib import Path

# A group ends at a line holding only `/`, only `&`, or `&end` / `$end` (all three styles occur in V4r4 files).
_GROUP = re.compile(r"&(\w+)\b(.*?)(?:^\s*/\s*$|^\s*&\s*$|^\s*[&$]end\b)", re.S | re.M | re.I)
_TOKEN = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|[^\s,']+")
_KEY = re.compile(r"^([A-Za-z_]\w*)\s*(\([^)]*\))?\s*=")


def _strip_comments(text):
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _convert(tok):
    if tok[0] in "'\"":
        q = tok[0]
        return tok[1:-1].replace(q + q, q)
    low = tok.lower()
    if low in (".true.", "t", ".t.", "true"):
        return True
    if low in (".false.", "f", ".f.", "false"):
        return False
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(low.replace("d", "e"))
    except ValueError:
        raise ValueError(f"cannot parse namelist value {tok!r}") from None


def _values(text):
    out = []
    for tok in _TOKEN.findall(text):
        m = re.fullmatch(r"(\d+)\*(.+)", tok)
        if m and tok[0] not in "'\"":
            out.extend([_convert(m.group(2))] * int(m.group(1)))
        else:
            out.append(_convert(tok))
    return out


def _assignments(body):
    """Split a group body into (key, value text), respecting quoted strings."""
    # positions of `key =` / `key(idx) =` outside strings
    starts = []
    i, n, in_str = 0, len(body), None
    while i < n:
        c = body[i]
        if in_str:
            if c == in_str:
                if i + 1 < n and body[i + 1] == in_str:
                    i += 2
                    continue
                in_str = None
        elif c in "'\"":
            in_str = c
        elif (c.isalpha() or c == "_") and (i == 0 or not (body[i - 1].isalnum() or body[i - 1] in "_.")):
            m = _KEY.match(body[i:])
            if m:
                starts.append((i, m))
                i += m.end()
                continue
        i += 1
    for k, (pos, m) in enumerate(starts):
        end = starts[k + 1][0] if k + 1 < len(starts) else n
        key = m.group(1).lower() + (re.sub(r"\s+", "", m.group(2)) if m.group(2) else "")
        yield key, body[pos + m.end():end]


def parse_namelist(text):
    groups = {}
    for gm in _GROUP.finditer(_strip_comments(text)):
        entries = groups.setdefault(gm.group(1).lower(), {})
        for key, vtext in _assignments(gm.group(2)):
            entries[key] = _values(vtext)
    return groups


def read_namelist(path):
    return parse_namelist(Path(path).read_text(errors="replace"))
