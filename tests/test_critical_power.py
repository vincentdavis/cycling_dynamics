"""Test for the criticalPower module"""

from src.cycling_dynamics.critical_power import CriticalPower


def test_convert_user_critical_power() -> None:
    """Test the conversion of user input to a dataframe and dictionary
    This happens on class initiation if the user provides a critical power profile."""
    user_input = """1, 1000
    5, 800
    30, 500
    60, 450
    300, 400
    1200, 350"""
    profile = user_input.split("\n")
    profile = [x.split(",") for x in profile]
    profile = {int(x[0]): int(x[1]) for x in profile}
    cpp = CriticalPower(cp_user=profile)
    df = cpp.cp_defined_df
    cp = cpp.cp_defined_dict
    # df, cp = convert_user_critical_power(profile)
    assert {1: 1000, 5: 800, 30: 500, 60: 450, 300: 400, 1200: 350}.items() <= cp.items()
    assert df["seconds"].max() == 1200
    assert df["power"].max() == 1000
    assert df["seconds"].min() == 1
    assert df["power"].min() == 350


def test_ramp_test_activity() -> None:
    user_input = """1, 1000
    5, 800
    30, 500
    60, 450
    300, 400
    1200, 350"""
    profile = user_input.split("\n")
    profile = [x.split(",") for x in profile]
    profile = {int(x[0]): int(x[1]) for x in profile}
    cpp = CriticalPower(cp_user=profile)
    df, dfwko = cpp.ramp_test_activity()
    assert df["power"].max() == 1000
    assert df["power"].min() == 350
    assert dfwko["power"].max() == 1000


def test_critical_power_read_fit() -> None:
    """Test the reading of a fit file"""
    FIT_FILE = "test_data/vincent_lap_1_24HOP_14012433014_ACTIVITY.fit"
    cpp = CriticalPower(activity=FIT_FILE)
    cpp.calculate_cp()
    correct_values = [
        (1, 680.0),
        (5, 550.2),
        (10, 515.2),
        (20, 446.05),
        (30, 397.7),
        (60, 331.3666666666667),
        (120, 280.34166666666664),
        (300, 261.0833333333333),
        (600, 258.47),
        (1200, 246.69916666666666),
    ]
    for point in correct_values:
        assert cpp.cp_points[point[0]].cp == point[1]


def test_critical_power_cp_intensity() -> None:
    FIT_FILE = "test_data/vincent_lap_1_24HOP_14012433014_ACTIVITY.fit"
    cpp = CriticalPower(activity=FIT_FILE)
    cpp.cp_intensity()


def test_ramp_test_activity():
    user_input = """1, 1000
    5, 800
    30, 500
    60, 450
    300, 400
    1200, 350"""
    profile = user_input.split("\n")
    profile = [x.split(",") for x in profile]
    profile = {int(x[0]): int(x[1]) for x in profile}
    cpp = CriticalPower(cp_user=profile)
    df, df_wko = cpp.ramp_test_activity()
    assert df["power"].max() == 1000
    assert df["power"].min() == 350
    assert df_wko["power"].max() == 1000


def test_make_zwo_from_ramp():
    user_input = """1, 1000
    5, 800
    30, 500
    60, 450
    300, 400
    1200, 350"""
    profile = user_input.split("\n")
    profile = [x.split(",") for x in profile]
    profile = {int(x[0]): int(x[1]) for x in profile}
    cpp = CriticalPower(cp_user=profile)
    df, df_wko = cpp.ramp_test_activity()
    wko1 = cpp.make_zwo_from_ramp(df_wko, filename=None, name="test", ftp=250)
    assert wko1 is not None
    assert wko1.startswith("<?xml version='1.0' encoding='UTF-8'?>")
    assert wko1.endswith("</workout_file>\n")
