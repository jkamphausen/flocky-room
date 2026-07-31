from src.tracker import UNKNOWN_LABEL, Tracker


def make_tracker(**overrides: object) -> Tracker:
    options: dict[str, object] = {
        "iou_threshold": 0.3,
        "max_age_frames": 2,
        "vote_window": 3,
        "recognition_threshold": 0.45,
        "reid_interval": 2,
    }
    options.update(overrides)
    return Tracker(**options)  # type: ignore[arg-type]


def test_iou_matching_preserves_track_id_and_expires_old_track() -> None:
    tracker = make_tracker()
    first = tracker.update([(0.0, 0.0, 10.0, 10.0)])[0]
    second = tracker.update([(1.0, 0.0, 11.0, 10.0)])[0]

    assert second.track_id == first.track_id
    assert second.age == 2
    tracker.update([])
    tracker.update([])
    assert len(tracker.tracks) == 1
    tracker.update([])
    assert tracker.tracks == ()


def test_recognition_is_due_for_new_tracks_then_at_interval() -> None:
    tracker = make_tracker(reid_interval=2)
    track = tracker.update([(0.0, 0.0, 10.0, 10.0)])[0]
    assert [item.track_id for item in tracker.tracks_needing_recognition()] == [track.track_id]

    tracker.add_recognition(track.track_id, "ada", 0.9)
    assert tracker.tracks_needing_recognition() == ()
    tracker.update([(0.0, 0.0, 10.0, 10.0)])
    assert tracker.tracks_needing_recognition() == ()
    tracker.update([(0.0, 0.0, 10.0, 10.0)])
    assert [item.track_id for item in tracker.tracks_needing_recognition()] == [track.track_id]


def test_vote_uses_majority_and_requires_mean_similarity_threshold() -> None:
    tracker = make_tracker()
    track = tracker.update([(0.0, 0.0, 10.0, 10.0)])[0]
    tracker.add_recognition(track.track_id, "ada", 0.7)
    tracker.add_recognition(track.track_id, "bert", 0.9)
    tracker.add_recognition(track.track_id, "ada", 0.1)
    assert track.label == UNKNOWN_LABEL

    tracker.add_recognition(track.track_id, "ada", 0.9)
    assert track.label == "ada"
    assert track.label_similarity == 0.5


def test_duplicate_person_label_keeps_highest_similarity_and_uses_fallback() -> None:
    tracker = make_tracker(vote_window=3)
    first, second = tracker.update(
        [(0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)]
    )
    tracker.add_recognition(first.track_id, "ada", 0.7)
    tracker.add_recognition(first.track_id, "bert", 0.8)
    tracker.add_recognition(first.track_id, "ada", 0.7)
    tracker.add_recognition(second.track_id, "ada", 0.95)

    assert second.label == "ada"
    assert first.label == "bert"
    assert len({track.label for track in tracker.tracks if track.label != UNKNOWN_LABEL}) == 2


def test_duplicate_person_label_without_second_candidate_becomes_unknown() -> None:
    tracker = make_tracker()
    first, second = tracker.update(
        [(0.0, 0.0, 10.0, 10.0), (20.0, 0.0, 30.0, 10.0)]
    )
    tracker.add_recognition(first.track_id, "ada", 0.7)
    tracker.add_recognition(second.track_id, "ada", 0.9)

    assert second.label == "ada"
    assert first.label == UNKNOWN_LABEL
