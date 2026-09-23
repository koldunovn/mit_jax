"""%MON parser: blocks keyed by time_tsnumber, cg2d lines attached to their step, relative comparison."""

from mitgcm_jax.io.monitor import compare_monitors, read_monitor

TEXT = """\
(PID.TID 0000.0001) %MON time_tsnumber                =                     1
(PID.TID 0000.0001) %MON dynstat_theta_mean           =   3.6036633347851E+00
(PID.TID 0000.0001)      cg2d_init_res =   1.56799778272944E+01
(PID.TID 0000.0001) %MON time_tsnumber                =                     2
(PID.TID 0000.0001) %MON dynstat_theta_mean           =   3.6036652180230E+00
(PID.TID 0000.0001) %MON dynstat_sst_max              =   3.2393200159235E+01
"""


def test_parse_and_compare(tmp_path):
    p = tmp_path / "STDOUT.0000"
    p.write_text(TEXT)
    m = read_monitor(p)
    assert sorted(m) == [1, 2]
    assert m[1]["dynstat_theta_mean"] == 3.6036633347851 and m[1]["cg2d_init_res"] == 15.6799778272944
    q = tmp_path / "other"
    q.write_text(TEXT.replace("3.6036652180230E+00", "3.6036652180231E+00").replace(
        "(PID.TID 0000.0001) %MON dynstat_sst_max              =   3.2393200159235E+01\n", ""))
    d, only = compare_monitors(m, read_monitor(q))
    assert d[(1, "dynstat_theta_mean")] == 0.0
    assert 2e-14 < d[(2, "dynstat_theta_mean")] < 3e-14
    assert only == ["dynstat_sst_max"]
