"""Test for the criticalPower module"""

from src.cycling_dynamics.critical_power import convert_user_critical_power


def test_convert_user_critical_power() -> None:
    user_input = """1, 1000
    5, 800
    30, 500
    60, 450
    300, 400
    1200, 350"""
    profile = user_input.split("\n")
    profile = [x.split(",") for x in profile]
    profile = {int(x[0]): int(x[1]) for x in profile}
    df, cp = convert_user_critical_power(profile)
    assert {1: 1000, 5: 800, 30: 500, 60: 450, 300: 400, 1200: 350}.items() <= cp.items()
    assert df["seconds"].max() == 1200
    assert df["power"].max() == 1000
    assert df["seconds"].min() == 1
    assert df["power"].min() == 350
