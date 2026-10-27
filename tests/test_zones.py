from zones import Zone, find_zone, load_zones, save_zones

SQUARE = [[100, 100], [200, 100], [200, 200], [100, 200]]
TRIANGLE = [[300, 300], [400, 300], [300, 400]]


def test_contains_polygon_and_edges():
    tri = Zone("bed", "resting", TRIANGLE)
    assert tri.contains((310, 310))
    assert tri.contains((300, 350))      # on the edge counts
    assert not tri.contains((390, 390))  # inside the bounding box, outside the triangle


def test_find_zone_returns_first_match_or_none():
    zones = [Zone("a", "x", SQUARE), Zone("b", "y", TRIANGLE)]
    assert find_zone(zones, (150, 150)).name == "a"
    assert find_zone(zones, (320, 320)).name == "b"
    assert find_zone(zones, (0, 0)) is None


def test_roundtrip_and_rescale(tmp_path):
    path = tmp_path / "zones.json"
    save_zones(path, [Zone("food", "eating", SQUARE)], (1920, 1080))

    same = load_zones(path, frame_size=(1920, 1080))
    assert same[0].points == SQUARE and same[0].still_label == "eating"

    # 640x360 stream: points scale by 1/3.
    small = load_zones(path, frame_size=(640, 360))
    assert small[0].points[0] == [33, 33]
    assert small[0].contains((50, 50))


def test_project_zones_file_loads():
    zones = load_zones("zones.json", frame_size=(1920, 1080))
    assert [z.name for z in zones] == ["food zone", "bed zone", "sofa zone"]
