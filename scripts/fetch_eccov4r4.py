#!/usr/bin/env python3
"""Fetch ECCO v4r4 inputs and reference products from PO.DAAC (Earthdata login from ~/.netrc).

Stdlib only. Files land under DATA_ROOT/<group>/ (DATA_ROOT = $MITJAX_DATA, mitgcm_jax/paths.py); nothing is ever
deleted. A download goes to `<name>.part` and is renamed only after its checksum matches the one PO.DAAC publishes
(sha512 sidecar for the ancillary archives, the CMR checksum for product granules). A file already present is skipped only if its checksum matches — never by name.
Interrupted downloads resume from the `.part` size (HTTP Range).

    fetch_eccov4r4.py list  GROUP...            show what a group would fetch (CMR query, no download)
    fetch_eccov4r4.py fetch GROUP...            download and verify
    fetch_eccov4r4.py groups                    list the groups
    fetch_eccov4r4.py extract ARCHIVE [--years 1992 ...] [--exclude GLOB ...] [--dest DIR]
                                                one streaming pass: write ARCHIVE.index.txt (every member), unpack
                                                every member except yearly files (name ending _YYYY) of other years
                                                and --exclude matches; never overwrites; sha256 manifest of output

Groups are defined in GROUPS below. The two forcing archives (196 GB, 94 GB) and data_constraints (9.6 GB) are only
published whole.
"""

import argparse
import fnmatch
import hashlib
import importlib.util
import re
import tarfile
import http.cookiejar
import json
import netrc
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _paths():
    """mitgcm_jax/paths.py loaded by file (importing the mitgcm_jax package would import jax)."""
    spec = importlib.util.spec_from_file_location("mitgcm_jax_paths",
                                                  Path(__file__).resolve().parents[1] / "mitgcm_jax" / "paths.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


DATA_ROOT = _paths().DATA                     # $MITJAX_DATA (default: <work root>/data/eccov4r4)
CMR = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
URS = "urs.earthdata.nasa.gov"
UA = "mitgcm-jax-fetch/0.1 (python urllib)"
ANCILLARY = "ECCO_L4_ANCILLARY_DATA_V4R4"

# group -> list of (collection short name, granule filter). Filter: {"granule": exact GranuleUR} or
# {"temporal": "start,end"} (CMR temporal, ISO 8601).
GROUPS = {
    # small ancillary archives needed by every run
    "ancillary_small": [
        (ANCILLARY, {"granule": "ancillary_data_native_grid_files_ECCO_V4r4"}),
        (ANCILLARY, {"granule": "ancillary_data_input_init_ECCO_V4r4"}),
        (ANCILLARY, {"granule": "ancillary_data_doc_ECCO_V4r4"}),
        (ANCILLARY, {"granule": "ancillary_data_misc_ECCO_V4r4"}),
    ],
    # full V4r4 forcing (adjusted + unadjusted + other + control_weights), published only as one archive
    "input_forcing": [(ANCILLARY, {"granule": "ancillary_data_input_forcing_ECCO_V4r4"})],
    # flux-forced forcing, published only as one archive
    "flux_forcing": [(ANCILLARY, {"granule": "ancillary_data_atm_flux_forcing_experiments_ECCO_V4r4"})],
    # cost-function inputs (observations, weights, sigma files for pkg/ecco and pkg/profiles), one archive (8.9 GiB)
    "data_constraints": [(ANCILLARY, {"granule": "ancillary_data_data_constraints_ECCO_V4r4"})],
    # geometry and final mixing coefficients (time-invariant)
    "products_fixed": [
        ("ECCO_L4_GEOMETRY_LLC0090GRID_V4R4", {}),
        ("ECCO_L4_OCEAN_3D_MIX_COEFFS_LLC0090GRID_V4R4", {}),
    ],
    # first instantaneous snapshot (1992-01-02T00, 11 steps into the run)
    "products_snap_19920102": [
        (c, {"temporal": "1992-01-02T00:00:00Z,1992-01-02T00:00:00Z"})
        for c in ("ECCO_L4_TEMP_SALINITY_LLC0090GRID_SNAPSHOT_V4R4", "ECCO_L4_SSH_LLC0090GRID_SNAPSHOT_V4R4",
                  "ECCO_L4_OBP_LLC0090GRID_SNAPSHOT_V4R4", "ECCO_L4_SEA_ICE_CONC_THICKNESS_LLC0090GRID_SNAPSHOT_V4R4",
                  "ECCO_L4_SEA_ICE_VELOCITY_LLC0090GRID_SNAPSHOT_V4R4")
    ],
    # 1992 monthly means
    "products_monthly_1992": [
        (c, {"temporal": "1992-01-01T00:00:00Z,1992-12-31T23:59:59Z"})
        for c in ("ECCO_L4_TEMP_SALINITY_LLC0090GRID_MONTHLY_V4R4", "ECCO_L4_SSH_LLC0090GRID_MONTHLY_V4R4",
                  "ECCO_L4_OCEAN_VEL_LLC0090GRID_MONTHLY_V4R4", "ECCO_L4_MIXED_LAYER_DEPTH_LLC0090GRID_MONTHLY_V4R4",
                  "ECCO_L4_SEA_ICE_CONC_THICKNESS_LLC0090GRID_MONTHLY_V4R4")
    ],
}


def _ssl_context():
    # The mambaforge base CA bundle is broken on Levante (curl exit 77 too); the system bundle works.
    for cafile in (os.environ.get("SSL_CERT_FILE"), "/etc/ssl/certs/ca-bundle.crt"):
        if cafile and Path(cafile).exists():
            return ssl.create_default_context(cafile=cafile)
    return ssl.create_default_context()


def opener():
    login, _, password = netrc.netrc().authenticators(URS)
    pw = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    pw.add_password(None, f"https://{URS}", login, password)
    op = urllib.request.build_opener(
        urllib.request.HTTPBasicAuthHandler(pw),
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
        urllib.request.HTTPSHandler(context=_ssl_context()))
    op.addheaders = [("User-Agent", UA)]
    return op


def cmr_granules(short_name, flt):
    """[(name, url, size_bytes, (algorithm, checksum) or None)] for data files of matching granules."""
    params = {"short_name": short_name, "provider": "POCLOUD", "page_size": "2000"}
    if "granule" in flt:
        params["granule_ur"] = flt["granule"]
    if "temporal" in flt:
        params["temporal"] = flt["temporal"]
    url = CMR + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    d = json.load(urllib.request.urlopen(req, timeout=120, context=_ssl_context()))
    out = []
    for it in d["items"]:
        umm = it["umm"]
        infos = {a["Name"]: a for a in umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation", [])}
        for ru in umm.get("RelatedUrls", []):
            if ru.get("Type") != "GET DATA" or not ru["URL"].startswith("https://"):
                continue
            name = ru["URL"].rsplit("/", 1)[1]
            if name.endswith((".sha512", ".md5")):
                continue
            info = infos.get(name, {})
            size = info.get("SizeInBytes")
            if size is None and info.get("Size") is not None:  # approximate: progress display only
                size = int(info["Size"] * {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[info.get("SizeUnit", "B")])
            ck = info.get("Checksum")
            out.append((name, ru["URL"], size, (ck["Algorithm"], ck["Value"]) if ck else None))
    return out


def _hash(path, algorithm):
    h = hashlib.new(algorithm.replace("-", "").lower())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_checksum(op, name, url, cmr_ck, size):
    """(algorithm, hex) to verify against, in order of preference: a .sha512 or .md5 sidecar, the CMR checksum,
    else ("size", exact byte count from CMR) — PO.DAAC publishes no checksum for the ECCO netCDF products."""
    for alg in ("sha512", "md5"):
        try:
            with op.open(url + "." + alg, timeout=120) as r:
                return (alg, r.read().decode().split()[0].strip().lower())
        except urllib.error.HTTPError as e:
            if e.code not in (403, 404):
                raise
    if cmr_ck:
        return (cmr_ck[0].replace("-", "").lower(), cmr_ck[1].lower())
    if size:
        return ("size", str(size))
    raise SystemExit(f"no checksum or size published for {name}; refusing to accept an unverifiable file")


def _verify(path, alg):
    return str(path.stat().st_size) if alg == "size" else _hash(path, alg)


def download(op, url, dest, size_hint, retries=50):
    """Stream `url` into `<dest>.part`, resuming via HTTP Range. A stream that ends early is NOT completion: PO.DAAC's
    signed redirect URLs expire (a 92 GiB transfer was cut at 36.6 GiB with no error), so every short read gets a
    fresh authenticated request from the current .part size. Completion = byte count equals the expected size."""
    part = dest.with_name(dest.name + ".part")
    expected = None  # only the server's Content-Range/Length decides (CMR sizes can be rounded MB); size_hint: display
    for attempt in range(retries):
        have = part.stat().st_size if part.exists() else 0
        if expected and have == expected:
            return part
        if expected and have > expected:
            raise SystemExit(f"{part} is larger ({have}) than expected ({expected}); inspect by hand")
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            try:
                r = op.open(req, timeout=300)
            except urllib.error.HTTPError as e:
                if e.code == 416 and have:  # range starts at/after the end: nothing left to fetch
                    return part
                raise
            with r:
                if have and r.status != 206:
                    have = 0  # server ignored Range: restart this partial file from byte 0
                cr = r.headers.get("Content-Range")  # "bytes a-b/total"
                if cr and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
                    expected = int(cr.rsplit("/", 1)[1])
                elif not have and r.headers.get("Content-Length"):
                    expected = int(r.headers["Content-Length"])
                mode = "ab" if have else "wb"
                t0, done = time.time(), have
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 22)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - t0 > 30:
                            rate = (done - have) / (time.time() - t0) / 2**20
                            print(f"    {dest.name}: {done / 2**30:.2f}/{(expected or size_hint or 0) / 2**30:.2f} GiB, "
                                  f"{rate:.1f} MiB/s", flush=True)
                            t0, have = time.time(), done
            if expected is None or done == expected:
                return part
            print(f"    stream ended at {done} of {expected} bytes; resuming (attempt {attempt + 1})", flush=True)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            print(f"    attempt {attempt + 1} failed: {e!r}; retrying in 30 s", flush=True)
            time.sleep(30)
    raise SystemExit(f"download incomplete after {retries} attempts: {url}")


def _record(outdir, dest, alg, want):
    """Own record of what we accepted (keep-own-copies rule): sha256 plus the published value, once per file."""
    man = outdir / "MANIFEST.sha256"
    rel = str(dest.relative_to(outdir))
    if man.exists() and any(line.split()[1] == rel for line in man.read_text().splitlines() if line.strip()):
        return
    with open(man, "a") as m:
        m.write(f"{_hash(dest, 'sha256')}  {rel}  [{alg} {want}]\n")


def fetch(group, dry=False):
    op = None if dry else opener()
    outdir = DATA_ROOT / group
    items = [(sn, g) for sn, flt in GROUPS[group] for g in cmr_granules(sn, flt)]
    if not items:
        raise SystemExit(f"{group}: CMR returned no granules — check the filters")
    total = sum(g[2] or 0 for _, g in items)
    print(f"{group}: {len(items)} files, {total / 2**30:.2f} GiB -> {outdir}")
    for sn, (name, url, size, ck) in items:
        sub = outdir if sn == ANCILLARY else outdir / sn
        dest = sub / name
        print(f"  {name} ({(size or 0) / 2**20:.1f} MiB)")
        if dry:
            continue
        sub.mkdir(parents=True, exist_ok=True)
        alg, want = expected_checksum(op, name, url, ck, size)
        if dest.exists():
            if _verify(dest, alg) == want:
                print("    present, checksum OK")
                _record(outdir, dest, alg, want)
                continue
            raise SystemExit(f"{dest} exists with a WRONG checksum; move it aside by hand (this script never deletes)")
        part = download(op, url, dest, size)
        got = _verify(part, alg)
        if got != want:
            raise SystemExit(f"checksum mismatch for {part}: {alg} {got} != {want} (partial file kept)")
        part.rename(dest)
        _record(outdir, dest, alg, want)
        print(f"    OK {alg}")


def extract(archive, dest, years, excludes):
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    index = archive.with_name(archive.name + ".index.txt")
    man = dest / "MANIFEST.extracted.sha256"
    n_all = n_out = 0
    with tarfile.open(archive, "r|gz") as tf, open(index, "w") as idx:
        for m in tf:
            idx.write(f"{m.size:>14d}  {'d' if m.isdir() else 'f' if m.isfile() else '?'}  {m.name}\n")
            n_all += 1
            if not m.isfile():
                continue
            ym = re.search(r"_(\d{4})$", m.name)
            if ym and int(ym.group(1)) not in years:
                continue
            if any(fnmatch.fnmatch(m.name, g) for g in excludes):
                continue
            target = (dest / m.name).resolve()
            if root not in target.parents:
                raise SystemExit(f"refusing member outside {dest}: {m.name}")
            if target.exists():  # never overwrite: accept only if byte-identical to the member
                h = hashlib.sha256()
                with tf.extractfile(m) as src:
                    for chunk in iter(lambda: src.read(1 << 24), b""):
                        h.update(chunk)
                if _hash(target, "sha256") != h.hexdigest():
                    raise SystemExit(f"{target} exists and differs from the archive member; not overwriting")
                rel = str(target.relative_to(root))
                if not (man.exists() and any(l.split()[1] == rel for l in man.read_text().splitlines() if l)):
                    with open(man, "a") as f:
                        f.write(f"{h.hexdigest()}  {rel}\n")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            h = hashlib.sha256()
            with tf.extractfile(m) as src, open(part, "wb") as out:
                for chunk in iter(lambda: src.read(1 << 24), b""):
                    out.write(chunk)
                    h.update(chunk)
            part.rename(target)
            with open(man, "a") as f:
                f.write(f"{h.hexdigest()}  {target.relative_to(root)}\n")
            n_out += 1
            print(f"  {m.name} ({m.size / 2**20:.1f} MiB)", flush=True)
    print(f"{archive.name}: {n_all} members indexed in {index.name}, {n_out} extracted to {dest}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("action", choices=("list", "fetch", "groups", "extract"))
    ap.add_argument("groups", nargs="*", help="group names, or the archive path for extract")
    ap.add_argument("--years", nargs="*", type=int, default=[1992])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--dest", default=str(DATA_ROOT))
    a = ap.parse_args(argv)
    if a.action == "extract":
        for arch in a.groups:
            extract(arch, a.dest, set(a.years), a.exclude)
        return 0
    if a.action == "groups":
        for g, spec in GROUPS.items():
            print(g, [s for s, _ in spec])
        return 0
    for g in a.groups:
        if g not in GROUPS:
            raise SystemExit(f"unknown group {g}; known: {list(GROUPS)}")
        fetch(g, dry=(a.action == "list"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
