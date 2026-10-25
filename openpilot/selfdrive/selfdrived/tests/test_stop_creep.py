from openpilot.selfdrive.selfdrived.stop_creep import (StopCreepWarning, NEAR_REST_V, CREEP_V,
                                                      CREEP_T)

DT = 0.01


def feed(w, steps, v, *, enabled=True, should_stop=True, gas=False, brake=False):
  return [w.update(enabled, should_stop, v, gas, brake) for _ in range(int(round(steps / DT)))]


def test_6db_crawl_fires_one_second_after_the_roll_starts():
  # 6db-like: 0.35 -> 0.03, held at 0.03, then rising to 0.30 and held there
  w = StopCreepWarning(DT)
  out = []
  out += feed(w, 1.0, 0.35)
  out += feed(w, 0.6, 0.03)
  out += feed(w, 0.1, 0.10)                   # still below CREEP_V
  crossing = len(out)                         # index of the first sample above CREEP_V
  out += feed(w, 1.4, 0.25)
  out += feed(w, 3.0, 0.30)
  fired = [i for i, hit in enumerate(out) if hit]
  assert fired, "the crawl never produced a warning"
  # the warning must land one CREEP_T after the crossing, not before
  assert fired[0] == crossing + int(round(CREEP_T / DT)) - 1, f"onset at sample {fired[0]}, crossing {crossing}"


def test_normal_stop_never_warns():
  w = StopCreepWarning(DT)
  out = feed(w, 0.5, 0.35) + feed(w, 0.3, 0.10) + feed(w, 0.2, 0.02) + feed(w, 5.0, 0.0)
  assert not any(out)


def test_departure_never_warns():
  w = StopCreepWarning(DT)
  out = feed(w, 0.3, 0.30) + feed(w, 0.3, 0.03) + feed(w, 0.5, 0.02)
  out += feed(w, 2.0, 0.05, should_stop=False) + feed(w, 3.0, 0.30, should_stop=False)
  assert not any(out)


def test_gas_or_brake_resets_the_timer():
  w = StopCreepWarning(DT)
  out = feed(w, 0.3, 0.35) + feed(w, 0.2, 0.02)
  # 2 s of creep past the threshold with a pedal down must stay silent
  out += feed(w, 2.0, 0.30, gas=True)
  out += feed(w, 2.0, 0.30, brake=True)
  out += feed(w, 0.1, 0.02)                   # stop again, re-arm near rest
  # pedal released: the timer restarts from zero
  out += feed(w, CREEP_T - 0.05, 0.30)
  early = out[-int(round((CREEP_T - 0.05) / DT)):]
  assert not any(early), "the timer survived the pedal press"
  assert any(feed(w, 0.1, 0.30)), "the timer did not restart after the release"


def test_jitter_is_not_a_continuous_creep():
  w = StopCreepWarning(DT)
  out = feed(w, 0.2, 0.02)                    # come to rest first so the creep is reachable
  for _ in range(3):
    out += feed(w, 0.3, 0.18)
    out += feed(w, 0.3, 0.22)
  assert not any(out)


def test_no_near_rest_never_warns():
  assert not any(feed(StopCreepWarning(DT), 5.0, 0.5))


def test_disabled_never_warns():
  w = StopCreepWarning(DT)
  out = feed(w, 0.5, 0.35, enabled=False) + feed(w, 5.0, 0.30, enabled=False)
  assert not any(out)
