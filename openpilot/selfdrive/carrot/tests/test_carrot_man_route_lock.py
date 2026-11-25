from __future__ import annotations

import threading
from types import SimpleNamespace

from openpilot.selfdrive.carrot.carrot_man import CarrotMan


class _SpyLock:
  def __init__(self, on_enter=None, on_exit=None):
    self._lock = threading.Lock()
    self._on_enter = on_enter
    self._on_exit = on_exit

  def __enter__(self):
    self._lock.acquire()
    if self._on_enter:
      self._on_enter()
    return self

  def __exit__(self, exc_type, exc, tb):
    try:
      if self._on_exit:
        self._on_exit()
    finally:
      self._lock.release()


def test_handle_route_updates_navd_active_atomically_under_navi_lock():
  cm = CarrotMan.__new__(CarrotMan)
  observed_inside_lock = []

  def on_exit():
    observed_inside_lock.append((cm.navi_points_active, cm.navd_active, len(cm.navi_points)))

  cm._navi_points_lock = _SpyLock(on_exit=on_exit)
  cm.navi_points = []
  cm.navi_points_start_index = 0
  cm.navi_points_active = False
  cm.navd_active = False
  cm.pm = SimpleNamespace(send=lambda *a, **k: None)
  cm.params = SimpleNamespace(put=lambda *a, **k: None, remove=lambda *a, **k: None)

  cm.handle_route([{"x": 127.0, "y": 37.5}, {"x": 127.01, "y": 37.51}])
  assert observed_inside_lock == [(True, True, 2)]

  cm.handle_route([])
  assert observed_inside_lock[-1] == (False, False, 0)


def test_carrot_serv_and_carrot_man_throttle_periodic_param_reads(monkeypatch):
  from openpilot.selfdrive.carrot.carrot_serv import CarrotServ

  clock = [100.0]
  monkeypatch.setattr("openpilot.selfdrive.carrot.carrot_serv.time.monotonic", lambda: clock[0])
  monkeypatch.setattr("openpilot.selfdrive.carrot.carrot_man.time.monotonic", lambda: clock[0])

  reads = []

  class FakeParams:
    def get_int(self, key):
      reads.append(key)
      return 10

    def get_float(self, key):
      reads.append(key)
      return 100.0

    def get_bool(self, key):
      reads.append(key)
      return False

    def get(self, key):
      reads.append(key)
      return "30" if key == "AutoCurveSpeedLowerLimit" else "main_ko"

  serv = CarrotServ.__new__(CarrotServ)
  serv.params = FakeParams()
  serv._params_poll_t = -1e9
  serv.lang = "en"

  serv.update_params(force=False)
  first_count = len(reads)
  assert first_count > 10
  # Repeated calls within 1.0s must not touch flash params
  serv.update_params(force=False)
  clock[0] += 0.5
  serv.update_params(force=False)
  assert len(reads) == first_count
  # After 1.0s, refresh happens once
  clock[0] += 0.6
  serv.update_params(force=False)
  assert len(reads) == first_count * 2
  # force=True always reloads immediately
  serv.update_params(force=True)
  assert len(reads) == first_count * 3

  cm = CarrotMan.__new__(CarrotMan)
  cm.params = FakeParams()
  reads.clear()
  cm.carrot_curve_speed_params(force=False)
  assert reads == ["AutoCurveSpeedFactor"]
  cm.carrot_curve_speed_params(force=False)
  assert reads == ["AutoCurveSpeedFactor"]
  clock[0] += 1.1
  cm.carrot_curve_speed_params(force=False)
  assert reads == ["AutoCurveSpeedFactor", "AutoCurveSpeedFactor"]


def test_xiaoge_broadcaster_caps_clients_and_drops_dead_client():
  from openpilot.selfdrive.carrot.xiaoge_data import MAX_CLIENTS, XiaogeDataBroadcaster

  b = XiaogeDataBroadcaster.__new__(XiaogeDataBroadcaster)
  b.clients = {}
  b.client_send_locks = {}
  b.clients_lock = threading.Lock()
  b.server_running = True

  closed = []

  class FakeSock:
    def __init__(self, fail_send=False):
      self.fail_send = fail_send
      self.timeout = None

    def settimeout(self, t):
      self.timeout = t

    def setsockopt(self, *args):
      pass

    def sendall(self, data):
      if self.fail_send:
        raise OSError("broken pipe")

    def close(self):
      closed.append(self)

  dead = FakeSock(fail_send=True)
  alive = FakeSock(fail_send=False)
  b.clients[("127.0.0.1", 1)] = dead
  b.client_send_locks[("127.0.0.1", 1)] = threading.Lock()
  b.clients[("127.0.0.1", 2)] = alive
  b.client_send_locks[("127.0.0.1", 2)] = threading.Lock()

  b.broadcast_to_clients(b"hello")
  assert ("127.0.0.1", 1) not in b.clients
  assert ("127.0.0.1", 1) not in b.client_send_locks
  assert ("127.0.0.1", 2) in b.clients
  assert dead in closed

  # Fill to MAX_CLIENTS and verify overflow client is rejected and closed.
  for i in range(3, MAX_CLIENTS + 2):
    b.clients[("127.0.0.1", i)] = FakeSock()
  overflow = FakeSock()
  b.handle_client(overflow, ("127.0.0.1", 9999))
  assert overflow in closed
  assert ("127.0.0.1", 9999) not in b.clients


def test_xiaoge_handle_client_preserves_partial_frame_across_socket_timeout():
  import socket
  import struct
  from openpilot.selfdrive.carrot.xiaoge_data import CLIENT_SEND_TIMEOUT_S, XiaogeDataBroadcaster

  b = XiaogeDataBroadcaster.__new__(XiaogeDataBroadcaster)
  b.clients = {}
  b.client_send_locks = {}
  b.clients_lock = threading.Lock()
  b.server_running = True

  sent = []
  ping = struct.pack("!I", 2)
  steps = iter([ping[:2], socket.timeout("idle"), ping[2:], b""])

  class PartialSock:
    def __init__(self):
      self.timeout = None

    def settimeout(self, t):
      self.timeout = t

    def setsockopt(self, *args):
      pass

    def recv(self, n):
      item = next(steps)
      if isinstance(item, BaseException):
        raise item
      assert n >= len(item)
      return item

    def sendall(self, data):
      sent.append(data)

    def close(self):
      pass

  sock = PartialSock()
  b.handle_client(sock, ("127.0.0.1", 5000))
  assert sock.timeout == CLIENT_SEND_TIMEOUT_S
  assert sent == [struct.pack("!I", 0)]


def test_recovery_start_quick_tunnel_handles_multiline_burst_without_timeout(monkeypatch):
  import sys
  from openpilot.selfdrive.carrot.recovery import server as recovery_server

  monkeypatch.delenv("CARROT_SUPPORT_FAKE_TUNNEL_URL", raising=False)
  monkeypatch.setattr(recovery_server, "_cloudflared_path", lambda: sys.executable)

  real_popen = recovery_server.subprocess.Popen

  def fake_popen(_cmd, **kwargs):
    # Emit two lines in a single write, then stay open until terminated.
    code = (
      "import sys, time; "
      "sys.stdout.buffer.write(b'starting cloudflared\\nhttps://demo-test.trycloudflare.com\\n'); "
      "sys.stdout.buffer.flush(); "
      "time.sleep(30)"
    )
    return real_popen([sys.executable, "-c", code], **kwargs)

  monkeypatch.setattr(recovery_server.subprocess, "Popen", fake_popen)

  tunnel = recovery_server._start_quick_tunnel("http://127.0.0.1:8080", timeout_s=3.0)
  try:
    assert tunnel.url == "https://demo-test.trycloudflare.com"
  finally:
    tunnel.stop()
