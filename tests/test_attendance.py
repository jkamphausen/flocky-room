from src.attendance import AttendanceRegistry


def test_update_sets_first_and_last_seen_without_changing_first_seen() -> None:
    registry = AttendanceRegistry(present_timeout_s=20.0)
    registry.update("ada", now=10.0)
    registry.update("ada", now=17.0)

    assert registry.records["ada"].first_seen == 10.0
    assert registry.records["ada"].last_seen == 17.0


def test_person_is_present_only_before_timeout_and_registry_can_be_cleared() -> None:
    registry = AttendanceRegistry(present_timeout_s=20.0)
    registry.update("ada", now=100.0)

    assert registry.is_present("ada", now=119.9)
    assert not registry.is_present("ada", now=120.0)
    assert not registry.is_present("unknown", now=100.0)
    assert registry.present_person_ids(now=110.0) == {"ada"}
    registry.clear()
    assert registry.records == {}
