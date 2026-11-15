from vdet.analytics import LineCounter, TrackRegistry


def make_counter():
    # horizontal line y=50 from x=0 to x=100, direction vector points +x
    return LineCounter((0, 50), (100, 50))


def test_crossing_down_counts_once():
    c = make_counter()
    for y in (10, 30, 49, 51, 70, 90):
        c.update(1, (50, y), "person")
    assert c.total_in + c.total_out == 1
    assert len(c.events) == 1


def test_direction_is_opposite_for_opposite_motion():
    c = make_counter()
    for y in (10, 90):
        c.update(1, (50, y), "car")
    for y in (90, 10):
        c.update(2, (50, y), "car")
    dirs = {e.track_id: e.direction for e in c.events}
    assert dirs[1] != dirs[2]
    assert c.counts["car"]["in"] == 1 and c.counts["car"]["out"] == 1


def test_jitter_on_line_does_not_double_count():
    c = make_counter()
    for y in (40, 60, 40, 60, 40, 60):
        c.update(7, (50, y), "person")
    # at most one count per direction for a single track
    assert c.total_in <= 1 and c.total_out <= 1


def test_no_count_when_passing_beside_segment():
    c = make_counter()
    for y in (10, 90):
        c.update(3, (150, y), "bus")  # x=150 is past the end of the segment
    assert c.total_in + c.total_out == 0


def test_no_count_without_crossing():
    c = make_counter()
    for y in (10, 20, 30, 40):
        c.update(4, (50, y), "person")
    assert not c.events


def test_registry_majority_vote_and_filtering():
    r = TrackRegistry(trail_length=5)
    for i, name in enumerate(["car", "truck", "car", "car"]):
        r.update(1, name, 0.8, (i * 10, 0), frame=i)
    r.update(2, "person", 0.4, (0, 0), frame=0)  # single-frame flicker
    s = r.summary(fps=10)
    assert len(s) == 1 and s[0]["class"] == "car"
    assert s[0]["path_px"] == 30.0
    assert r.unique_counts() == {"car": 1}
    assert len(r.tracks[1].trail) == 4
