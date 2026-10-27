#!/usr/bin/env python3
"""One-shot parked diagnostic dump for the Palisade ATC turn-signal feature.

Why: the stationary web test returned ack_timeout -> the ICU (0x778) never
answered the 0x770 diagnostic session sent on bus 0. We need, in a single car
visit, (1) the native turn-lever command/lamp frames on every bus and (2) which
bus/address the ICU actually answers UDS on. This dumps both and lets the PC do
the rest.

It writes a carrot_tlv_rec session (index.jsonl) that the existing log_sync
process auto-uploads to the PC as tlvrec--<ts>/ and then deletes locally. No
networking code here, and no change to the driving safety gate: Phase 2 uses the
stock elm327 diagnostic mode (same as query_fw_versions) only while openpilot is
stopped, and openpilot is always relaunched afterwards.

Run at the car, parked, ignition on:
    cd /data/openpilot && ./selfdrive/carrot/tools/atc_car_dump.py

Phase 1 (openpilot stays up): follow the on-screen lever prompts.
Phase 2 (openpilot stopped ~2-3 min, auto-restarted): UDS scan, hands-off.
"""
import json
import os
import sys
import time

BASEDIR = os.environ.get("BASEDIR", "/data/openpilot")
if BASEDIR not in sys.path:
  sys.path.insert(0, BASEDIR)

TLVREC_ROOT = "/data/media/0/carrot_tlv_rec"

# Phase 1 guided schedule: (seconds_into_phase, label). The user reads these
# prompts and operates the real lever; every CAN frame on every bus is recorded
# with a monotonic timestamp so the PC can slice frames by label.
SCHEDULE = [
  (0.0, "idle — do nothing"),
  (5.0, "LEFT lever ON (hold it)"),
  (12.0, "idle — lever OFF"),
  (17.0, "RIGHT lever ON (hold it)"),
  (24.0, "idle — lever OFF"),
  (29.0, "HAZARD ON"),
  (36.0, "HAZARD OFF — done"),
]
PHASE1_SECONDS = 41.0

# Diagnostic addresses to sweep (11-bit). 0x7df/func-broadcast skipped.
SCAN_ADDRS = [0x700 + i for i in range(256)]
SCAN_TIMEOUT = 0.1           # per-address reachability probe
PROBE_TIMEOUT = 0.5          # targeted 0x770 probe
ICU_TX = 0x770
ICU_RX = 0x778


def _now():
  return time.monotonic()


class Dump:
  def __init__(self):
    ts = time.strftime("%Y%m%d-%H%M%S")
    self.name = ts
    os.makedirs(TLVREC_ROOT, exist_ok=True)
    # Build under a non-matching name so log_sync never grabs a half-written
    # file; rename into the \d{8}-\d{6} pattern at the very end.
    self.tmp_dir = os.path.join(TLVREC_ROOT, f"atcdump.tmp-{ts}")
    self.final_dir = os.path.join(TLVREC_ROOT, ts)
    os.makedirs(self.tmp_dir, exist_ok=True)
    self.fp = open(os.path.join(self.tmp_dir, "index.jsonl"), "w", encoding="utf-8")
    self.t0 = _now()

  def rec(self, **kw):
    kw.setdefault("t", round(_now() - self.t0, 4))
    self.fp.write(json.dumps(kw, ensure_ascii=False) + "\n")

  def finish(self):
    try:
      self.fp.flush()
      os.fsync(self.fp.fileno())
    except OSError:
      pass
    self.fp.close()
    os.rename(self.tmp_dir, self.final_dir)
    return self.final_dir


def phase1(dump):
  """Record every CAN frame on every bus while the user works the lever."""
  from cereal import messaging

  print("\n=== PHASE 1: native lever capture (openpilot stays running) ===")
  print("Follow the prompts. Each step is ~5 s.\n")
  sm = messaging.SubMaster(["can", "carState"])

  # settle + grab one carState for the parked check / meta
  gear, v_ego = "", None
  deadline = _now() + 2.0
  while _now() < deadline:
    sm.update(100)
    if sm.updated["carState"]:
      cs = sm["carState"]
      gear = str(cs.gearShifter)
      v_ego = float(cs.vEgo)
      break
  dump.rec(kind="phase1_meta", gear=gear, vEgo=v_ego)

  start = _now()
  next_idx = 0
  last_print = -1.0
  frame_count = 0
  while True:
    rel = _now() - start
    if rel >= PHASE1_SECONDS:
      break
    # advance prompt
    if next_idx < len(SCHEDULE) and rel >= SCHEDULE[next_idx][0]:
      label = SCHEDULE[next_idx][1]
      dump.rec(kind="phase1_prompt", label=label)
      print(f"  [{rel:4.1f}s] >>> {label}")
      next_idx += 1
    # live countdown to next prompt
    if next_idx < len(SCHEDULE):
      remain = SCHEDULE[next_idx][0] - rel
      if remain >= 0 and int(remain) != int(last_print):
        last_print = remain
    sm.update(50)
    if sm.updated["can"]:
      for c in sm["can"]:
        dump.rec(kind="f", s=c.src, a=c.address, d=bytes(c.dat).hex())
        frame_count += 1
  print(f"  phase1 done: {frame_count} frames recorded.")
  return gear, v_ego


def _stop_openpilot():
  print("\nStopping openpilot to free the panda ...")
  os.system("tmux kill-session -t comma >/dev/null 2>&1")
  # wait for pandad to release the USB device
  from panda import Panda
  for _ in range(30):
    time.sleep(0.5)
    try:
      serials = Panda.list()
      if serials:
        p = Panda(serials[0])
        return p
    except Exception:
      continue
  raise RuntimeError("panda did not become available after stopping openpilot")


def _restart_openpilot():
  print("\nRelaunching openpilot ...")
  launch = os.path.join(BASEDIR, "launch_openpilot.sh")
  os.system(f"cd {BASEDIR} && tmux new -s comma -d \"bash -lc '{launch}'\" >/dev/null 2>&1")


def _scan_bus(dump, p, uds_mod, bus, label):
  UdsClient = uds_mod.UdsClient
  SESSION_TYPE = uds_mod.SESSION_TYPE
  DATA_IDENTIFIER_TYPE = uds_mod.DATA_IDENTIFIER_TYPE
  MessageTimeoutError = uds_mod.MessageTimeoutError
  NegativeResponseError = uds_mod.NegativeResponseError

  print(f"  scanning {label} (bus {bus}) 0x700-0x7ff ...")
  responders = []
  for addr in SCAN_ADDRS:
    if addr in (0x7df,):
      continue
    uds = UdsClient(p, addr, bus=bus, timeout=SCAN_TIMEOUT)
    present = False
    detail = None
    try:
      uds.tester_present()
      present, detail = True, "tester_present_ok"
    except NegativeResponseError as e:
      present, detail = True, f"negative:{e}"          # a NRC still proves an ECU is there
    except MessageTimeoutError:
      present = False
    except Exception as e:
      present, detail = False, f"err:{e}"
    if not present:
      continue
    vin = None
    try:
      uds.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC)
    except Exception:
      pass
    try:
      vin = uds.read_data_by_identifier(DATA_IDENTIFIER_TYPE.VIN)
      vin = bytes(vin).hex()
    except Exception:
      vin = None
    rec = {"addr": hex(addr), "bus": bus, "label": label, "detail": detail, "vin": vin}
    responders.append(rec)
    dump.rec(kind="scan_responder", **rec)
    print(f"    responder {hex(addr)} bus{bus} {detail} vin={vin}")
  return responders


def _sniff(p, seconds):
  """Collect all CAN traffic for a window (addr, bus, hex)."""
  out = []
  end = _now() + seconds
  while _now() < end:
    try:
      for addr, dat, src in p.can_recv():
        out.append((addr, src, bytes(dat).hex()))
    except Exception:
      break
  return out


def _probe_icu(dump, p, uds_mod):
  """Send the exact session request on each candidate bus and watch for 0x778
  on ANY bus, to learn where (if anywhere) the ICU answers."""
  UdsClient = uds_mod.UdsClient
  SESSION_TYPE = uds_mod.SESSION_TYPE
  print("  probing ICU 0x770 session on candidate buses ...")
  for bus in (0, 1, 2):
    p.can_clear(0xFFFF)
    uds = UdsClient(p, ICU_TX, bus=bus, timeout=PROBE_TIMEOUT)
    err = None
    try:
      uds.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC)
    except Exception as e:
      err = str(e)
    sniff = _sniff(p, 0.4)
    icu_replies = [x for x in sniff if x[0] == ICU_RX]
    dump.rec(kind="probe770", bus=bus, err=err,
             icu_reply_count=len(icu_replies),
             icu_replies=icu_replies[:10],
             sniff_addrs=sorted({(hex(a), s) for a, s, _ in sniff})[:60])
    print(f"    bus{bus}: err={err} 0x778_replies={len(icu_replies)}")


def phase2(dump, gear, v_ego):
  print("\n=== PHASE 2: UDS scan (openpilot will stop ~2-3 min) ===")
  parked = (v_ego is not None and v_ego < 0.3) and ("park" in gear.lower() or gear == "")
  if not parked:
    msg = f"NOT parked (gear={gear!r}, vEgo={v_ego}); skipping Phase 2 for safety."
    print("  " + msg)
    dump.rec(kind="phase2_skipped", reason=msg)
    return
  print("  Car is parked. Phase 2 will stop openpilot and run a diagnostic scan.")
  for n in range(8, 0, -1):
    print(f"    starting in {n}s ... (Ctrl-C to abort)", end="\r")
    time.sleep(1)
  print()

  p = None
  try:
    p = _stop_openpilot()
    from opendbc.car import uds as uds_mod
    from opendbc.car.structs import CarParams
    try:
      serial = p.get_serial()
    except Exception:
      serial = None
    has_obd = bool(p.has_obd())
    dump.rec(kind="phase2_meta", panda_serial=str(serial), has_obd=has_obd)
    print(f"  panda serial={serial} has_obd={has_obd}")

    # bus 0 (no OBD multiplexing): param 1 = no OBD
    p.set_safety_mode(CarParams.SafetyModel.elm327, 1)
    time.sleep(0.2)
    _scan_bus(dump, p, uds_mod, bus=0, label="bus0")

    # bus 1 via OBD-II multiplexing: param 0 = OBD on
    p.set_safety_mode(CarParams.SafetyModel.elm327, 0)
    time.sleep(0.2)
    obd_bus = 1 if has_obd else 0
    _scan_bus(dump, p, uds_mod, bus=obd_bus, label="obd_bus1")

    # targeted ICU 0x770 probe with all-bus sniff (OBD on so bus1 reachable)
    _probe_icu(dump, p, uds_mod)
    print("  Phase 2 done.")
  except Exception as e:
    dump.rec(kind="phase2_error", msg=str(e))
    print(f"  Phase 2 error: {e}")
  finally:
    try:
      if p is not None:
        p.set_safety_mode(CarParams.SafetyModel.silent)
        p.close()
    except Exception:
      pass
    _restart_openpilot()


def main():
  print("ATC car dump starting. Keep the car PARKED with ignition ON.")
  dump = Dump()
  dump.rec(kind="meta", version=1, started=time.strftime("%Y-%m-%d %H:%M:%S"),
           git=os.popen(f"cd {BASEDIR} && git rev-parse --short HEAD 2>/dev/null").read().strip())
  try:
    gear, v_ego = phase1(dump)
    phase2(dump, gear, v_ego)
  finally:
    path = dump.finish()
    print(f"\nDump written to {path}")
    print("log_sync will upload it to the PC as tlvrec--" + dump.name + "/ and delete the local copy.")


if __name__ == "__main__":
  main()
