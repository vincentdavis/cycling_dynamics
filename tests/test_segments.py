import pytest

from cycling_dynamics.load_data import load_fit_file
from cycling_dynamics.segments import match_segments

FIT_FILES = [
    "alex_24HOP_2024-02-17-185919-ELEMNT ROAM 3FF5-53-0.fit",
    "alex_24HOP_2024-02-17-185919-ELEMNT_ROAM_3FF5-53-0.fit",
    "alex_24HOP_2024-02-18-002255-ELEMNT ROAM 3FF5-54-0.fit",
    "alex_24HOP_2024-02-18-002255-ELEMNT_ROAM_3FF5-54-0.fit",
    "vincent_lap_1_24HOP_14012433014_ACTIVITY.fit",
    "vincent_lap_2_24HOP_14010180820_ACTIVITY.fit",
]
TEST_DATA_DIR = "test_data"


@pytest.fixture()
def get_tracks():
    """Load all the tacks for the test."""
    return [load_fit_file(f"{TEST_DATA_DIR}/{f}") for f in FIT_FILES]


def test_match_segments(get_tracks):
    """Basic tests for matching segments across tracks."""
    matched = match_segments(tracks=get_tracks, start_distance=5000, length=100000)
    print(min([m.distance.max() for m in matched]), max([m.distance.max() for m in matched]))
    assert abs(min([m.distance.max() for m in matched]) - max([m.distance.max() for m in matched])) <= 1
