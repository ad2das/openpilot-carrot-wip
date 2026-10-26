import pytest
from dataclasses import replace

from openpilot.selfdrive.controls.lib.atc_signal import (
  AtcRequest,
  Direction,
  EpisodeProducer,
  RequestArbiter,
  RequestGates,
  request_is_well_formed,
)


SECOND = 1_000_000_000
NOW = 10 * SECOND
HEALTHY = RequestGates(enabled=True, lateral_active=True, can_valid=True, model_fresh=True, navigation_fresh=True, driver_input_known=True)


def request(phase="turn", episode=1, source_id=1, stamp=NOW, direction=None, **kwargs):
  if direction is None:
    direction = Direction.LEFT if phase in ("turn", "fork_pending", "fork_active") else Direction.NONE
  return AtcRequest(source_id, episode, stamp, direction, phase, True, **kwargs)


class TestEpisodeProducer:
  def setup_method(self):
    self.producer = EpisodeProducer(source_id=1)
    self.now = NOW

  def update(self, token, advance=0, **kwargs):
    self.now += int(advance * SECOND)
    args = {"now_ns": self.now, "source_mono_time": self.now, "source_valid": True}
    args.update(kwargs)
    return self.producer.update(token, **args)

  def arm(self):
    assert (self.update("none").phase) == ("none")
    assert (self.update("none", advance=1).reason) == ("idle_ready")

  def test_startup_active_and_prepare_cannot_start_episode(self):
    for token in ("turn left", "turn left prepare", "fork left", "atc left", "turn right"):
      result = self.update(token, advance=0.1)
      assert ((result.direction, result.episode, result.phase)) == ((Direction.NONE, 0, "canceled"))

  def test_fresh_none_must_last_full_interval(self):
    self.update("none")
    self.update("none", advance=0.999999999)
    assert (self.update("turn left").episode) == (0)
    self.arm()
    assert (self.update("turn left").episode) == (1)

  def test_prepare_to_active_preserves_episode(self):
    self.arm()
    prepare = self.update("turn left prepare")
    active = self.update("turn left", advance=0.1)
    assert ((prepare.phase, prepare.direction, prepare.episode)) == (("prepare", Direction.NONE, 1))
    assert ((active.phase, active.direction, active.episode)) == (("turn", Direction.LEFT, 1))

  def test_every_supported_direction_token(self):
    for base in ("turn", "fork", "atc"):
      for side, direction in (("left", Direction.LEFT), ("right", Direction.RIGHT)):
        self.producer = EpisodeProducer(source_id=1)
        self.arm()
        prepare = self.update(f"{base} {side} prepare")
        active = self.update(f"{base} {side}")
        assert (prepare.direction) == (Direction.NONE)
        assert (active.direction) == (direction)
        assert (active.episode) == (prepare.episode)

  def test_cancel_remains_sticky_across_type_and_phase_changes(self):
    self.arm()
    self.update("fork left")
    for token in ("fork left canceled", "turn left prepare", "atc left", "fork left", "turn left"):
      result = self.update(token, advance=0.1)
      assert ((result.phase, result.direction, result.episode)) == (("canceled", Direction.NONE, 1))

  def test_direction_change_cancels_and_cannot_change_back(self):
    self.arm()
    self.update("turn left prepare")
    for token in ("fork right prepare", "turn right", "turn left"):
      assert (self.update(token).phase) == ("canceled")

  def test_disconnect_and_reconnect_do_not_create_or_revive_episode(self):
    self.arm()
    self.update("turn left")
    assert not (self.update("turn left", source_alive=False).valid)
    result = self.update("turn left", advance=0.1)
    assert ((result.episode, result.direction)) == ((1, Direction.NONE))
    self.arm()
    assert (self.update("turn right").episode) == (2)

  def test_invalid_or_stale_gap_breaks_idle(self):
    for invalid in ({"source_alive": False}, {"source_valid": False}, {"source_mono_time": 1}):
      self.producer = EpisodeProducer(source_id=1)
      self.update("none")
      self.update("none", advance=0.9, **invalid)
      self.update("none", advance=0.2)
      assert (self.update("turn left").episode) == (0)

  def test_unobserved_gap_breaks_idle_and_cancels_active(self):
    self.update("none")
    self.update("none", advance=3)
    assert (self.update("turn left").episode) == (0)
    self.arm()
    self.update("turn left")
    assert (self.update("turn left", advance=3).phase) == ("canceled")

  def test_stale_stamp_cannot_be_renewed_by_publication(self):
    self.arm()
    active = self.update("turn left")
    stale = self.update("turn left", advance=2.000000001, source_mono_time=active.source_mono_time)
    assert not (stale.valid)
    assert (stale.source_mono_time) == (active.source_mono_time)
    assert (self.update("turn left").phase) == ("canceled")

  def test_backward_receipt_and_clock_fail_closed(self):
    for override in ({"source_mono_time": NOW - 1}, {"now_ns": NOW - 1}, {"source_mono_time": NOW + 10 * SECOND}):
      self.producer = EpisodeProducer(source_id=1)
      self.arm()
      self.update("turn left")
      assert not (self.update("turn left", **override).valid)
      assert (self.update("turn left").direction) == (Direction.NONE)

  def test_invalid_tokens_are_not_idle_or_navigation(self):
    for token in ("", "none ", "NONE", " turn left", "turn left canceled prepare", "turn leftover", None, True):
      self.producer = EpisodeProducer(source_id=1)
      self.arm()
      assert not (self.update(token).valid)
      assert (self.update("turn left").episode) == (0)

  def test_straight_and_none_prepare_do_not_rearm(self):
    self.arm()
    self.update("turn left")
    for token in ("straight", "straight prepare", "none prepare"):
      self.update(token, advance=1)
      assert (self.update("turn left").direction) == (Direction.NONE)
    assert (self.producer.episode) == (1)

  def test_explicit_cancel_and_wrong_boolean_types(self):
    for override in ({"cancel": True}, {"cancel": 1}, {"source_alive": 1}, {"source_valid": 1}):
      self.producer = EpisodeProducer(source_id=1)
      self.arm()
      self.update("turn left")
      assert (self.update("turn left", **override).direction) == (Direction.NONE)
      assert (self.update("turn left").direction) == (Direction.NONE)

  def test_none_interval_rearms_exactly_one_new_episode(self):
    self.arm()
    first = self.update("turn left")
    self.update("none")
    assert (self.update("turn left", advance=0.5).direction) == (Direction.NONE)
    self.arm()
    second = self.update("fork left prepare")
    assert (second.episode) == (first.episode + 1)
    assert (self.update("turn left").episode) == (second.episode)

  def test_configuration_rejects_invalid_ranges(self):
    for duration in (False, 0, -1, float("nan"), float("inf"), 1e308, 1e-20):
      with pytest.raises(ValueError):
        EpisodeProducer(max_age_seconds=duration)
    for source_id in (True, 0, -1, 1 << 64, 1.0):
      with pytest.raises(ValueError):
        EpisodeProducer(source_id=source_id)


class TestRequestArbiter:
  def setup_method(self):
    self.arbiter = RequestArbiter()

  def update(self, req=None, gates=HEALTHY, now=NOW):
    return self.arbiter.update(request() if req is None else req, gates, now_ns=now)

  def active(self):
    self.update(request("none", episode=0))
    assert (self.update().direction) == (Direction.LEFT)

  def test_startup_and_restart_active_suppressed(self):
    assert (self.update().direction) == (Direction.NONE)
    assert (self.update(request("prepare")).direction) == (Direction.NONE)
    assert (self.update().direction) == (Direction.NONE)
    self.update(request("none"))
    assert (self.update().direction) == (Direction.NONE)
    assert (self.update(request(episode=2)).direction) == (Direction.LEFT)
    restarted = RequestArbiter()
    assert (restarted.update(request(episode=2), HEALTHY, now_ns=NOW).direction) == (Direction.NONE)

  def test_initial_genuine_prepare_is_baseline(self):
    assert (self.update(request("prepare")).phase) == ("prepare")
    assert (self.update().direction) == (Direction.LEFT)

  def test_synthetic_prepare_is_not_startup_baseline(self):
    assert (self.update(request("prepare", reason="helper_not_ready")).phase) == ("canceled")
    assert (self.update().direction) == (Direction.NONE)

  def test_synthetic_prepare_marker_survives_multiple_consumers(self):
    self.update(request("none", episode=0))
    waiting = self.update(request("prepare", reason="helper_not_ready"))
    assert ((waiting.phase, waiting.reason)) == (("prepare", "helper_not_ready"))
    downstream = RequestArbiter()
    assert (downstream.update(waiting, HEALTHY, now_ns=NOW).phase) == ("canceled")
    active = self.update()
    assert (active.direction) == (Direction.LEFT)
    assert (downstream.update(active, HEALTHY, now_ns=NOW).direction) == (Direction.NONE)

  def test_prepare_after_active_cancels_stickily(self):
    self.active()
    assert (self.update(request("prepare")).phase) == ("canceled")
    assert (self.update().direction) == (Direction.NONE)

  def test_all_live_gates_apply_before_old_episode_filter(self):
    changes = [
      {"enabled": False},
      {"lateral_active": False},
      {"can_valid": False},
      {"model_fresh": False},
      {"navigation_fresh": False},
      {"driver_input_known": False},
      {"driver_direction": Direction.LEFT},
      {"driver_direction": Direction.RIGHT},
      {"hazard": True},
      {"trailer_connected": True},
      {"opposite_steering_torque": True},
      {"explicit_cancel": True},
    ]
    for change in changes:
      self.arbiter = RequestArbiter()
      self.active()
      self.update(request("none", episode=0), replace(HEALTHY, **change))
      assert (self.update().direction) == (Direction.NONE)

  def test_old_source_with_live_stop_also_cancels(self):
    self.update(request("none", episode=0, source_id=2))
    assert (self.update(request(source_id=2)).direction) == (Direction.LEFT)
    self.update(request(source_id=1), replace(HEALTHY, hazard=True))
    assert (self.update(request(source_id=2)).direction) == (Direction.NONE)

  def test_new_source_active_requires_new_baseline_and_episode(self):
    self.active()
    assert (self.update(request(source_id=2)).direction) == (Direction.NONE)
    assert (self.update(request(source_id=1)).direction) == (Direction.NONE)
    self.update(request("none", source_id=2))
    assert (self.update(request(source_id=2)).direction) == (Direction.NONE)
    assert (self.update(request(episode=2, source_id=2)).direction) == (Direction.LEFT)

  def test_new_source_genuine_prepare_can_start_after_inactive_baseline(self):
    self.active()
    assert (self.update(request("prepare", source_id=2)).phase) == ("prepare")
    assert (self.update(request(source_id=2)).direction) == (Direction.LEFT)

  def test_invalid_future_identity_cannot_poison_high_water(self):
    invalid = [
      {"phase": "unknown"},
      {"direction": True},
      {"direction": 1.0},
      {"direction": 3},
      {"valid": 1},
      {"valid": False},
      {"episode": True},
      {"episode": 1 << 64},
      {"source_id": NOW + 1},
      {"source_id": True},
      {"stamp": NOW + 1},
      {"stamp": NOW - 3 * SECOND},
      {"phase": "prepare", "direction": Direction.LEFT},
    ]
    for change in invalid:
      self.arbiter = RequestArbiter()
      self.active()
      fields = {"episode": 100, "source_id": 100}
      fields.update(change)
      valid = fields.pop("valid", True)
      malformed = replace(request(**fields), valid=valid)
      self.update(malformed)
      assert ((self.arbiter.source_id, self.arbiter.episode)) == ((1, 1))
      self.update(request("none"))
      assert (self.update(request(episode=2)).direction) == (Direction.LEFT)

  def test_stale_model_or_navigation_does_not_advance_identity(self):
    for field in ("model_fresh", "navigation_fresh"):
      self.arbiter = RequestArbiter()
      self.active()
      self.update(request(source_id=100, episode=100), replace(HEALTHY, **{field: False}))
      assert ((self.arbiter.source_id, self.arbiter.episode)) == ((1, 1))

  def test_invalid_boolean_and_driver_states_fail_closed(self):
    for field in RequestGates.__dataclass_fields__:
      invalid = True if field == "driver_direction" else 1
      self.arbiter = RequestArbiter()
      self.active()
      assert (self.update(gates=replace(HEALTHY, **{field: invalid})).direction) == (Direction.NONE)
      assert (self.update().direction) == (Direction.NONE)

  def test_direction_reversal_is_sticky(self):
    self.active()
    assert (self.update(request(direction=Direction.RIGHT)).direction) == (Direction.NONE)
    assert (self.update().direction) == (Direction.NONE)

  def test_phase_changes_do_not_rearm_canceled_episode(self):
    self.active()
    self.update(gates=replace(HEALTHY, hazard=True))
    for phase in ("prepare", "fork_pending", "fork_active", "turn", "none", "turn"):
      assert (self.update(request(phase)).direction) == (Direction.NONE)

  def test_future_episode_needs_none_not_just_type_change(self):
    self.active()
    assert (self.update(request(episode=2)).direction) == (Direction.NONE)
    assert (self.update(request("prepare", episode=3)).direction) == (Direction.NONE)
    self.update(request("none", episode=3))
    assert (self.update(request(episode=4)).direction) == (Direction.LEFT)

  def test_stale_same_request_cancels_and_fresh_copy_cannot_revive(self):
    self.active()
    assert not (self.update(now=NOW + 2 * SECOND + 1).valid)
    assert (self.update(request(stamp=NOW + 3 * SECOND), now=NOW + 3 * SECOND).direction) == (Direction.NONE)

  def test_unobserved_consumer_gap_cancels_even_with_fresh_request(self):
    self.active()
    assert (self.update(request(stamp=NOW + 3 * SECOND), now=NOW + 3 * SECOND).direction) == (Direction.NONE)

  def test_backward_clock_cancels(self):
    self.active()
    assert not (self.update(now=NOW - 1).valid)
    assert (self.update().direction) == (Direction.NONE)

  def test_old_timestamp_is_discarded_without_moving_high_water(self):
    self.active()
    self.update(request(stamp=NOW + 1), now=NOW + 1)
    assert (self.update(request(episode=100, stamp=NOW), now=NOW + 1).direction) == (Direction.NONE)
    assert (self.arbiter.episode) == (1)
    assert (self.update(request(stamp=NOW + 1), now=NOW + 1).direction) == (Direction.LEFT)

  def test_malformed_object_and_all_defaults_fail_closed(self):
    self.active()
    assert not (self.update(object()).valid)
    assert (self.update(gates=RequestGates()).direction) == (Direction.NONE)


class TestPipeline:
  def test_real_instances_prepare_active_cancel_idle_next_episode(self):
    producer, model, control = EpisodeProducer(source_id=1), RequestArbiter(), RequestArbiter()

    def step(token, time, gates=HEALTHY):
      raw = producer.update(token, now_ns=time, source_mono_time=time, source_valid=True)
      middle = model.update(raw, gates, now_ns=time)
      return raw, middle, control.update(middle, gates, now_ns=time)

    step("none", NOW)
    step("none", NOW + SECOND)
    raw, _, output = step("fork left prepare", NOW + SECOND + 1)
    assert ((raw.episode, output.phase)) == ((1, "prepare"))
    _, _, output = step("fork left", NOW + SECOND + 2)
    assert (output.direction) == (Direction.LEFT)
    step("fork left", NOW + SECOND + 3, replace(HEALTHY, hazard=True))
    _, _, output = step("turn left", NOW + SECOND + 4)
    assert (output.direction) == (Direction.NONE)
    step("none", NOW + SECOND + 5)
    step("none", NOW + 2 * SECOND + 5)
    raw, _, output = step("turn right", NOW + 2 * SECOND + 6)
    assert ((raw.episode, output.direction, output.reason)) == ((2, Direction.RIGHT, "request_allowed"))

  def test_request_shape_does_not_claim_physical_output(self):
    assert (set(AtcRequest.__dataclass_fields__)) == ({"source_id", "episode", "source_mono_time", "direction", "phase", "valid", "reason"})
    assert request_is_well_formed(request("none", episode=0))
    assert request_is_well_formed(request("canceled", episode=0))
    assert not (request_is_well_formed(request("prepare", episode=0)))
