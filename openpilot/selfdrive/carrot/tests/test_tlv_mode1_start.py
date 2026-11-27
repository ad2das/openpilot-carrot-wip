"""R6: a traffic start must not launch a stopped car in mode 1 (stop only)."""
import os

from openpilot.selfdrive.carrot.carrot_functions import XState, tlv_start_xstate_ok

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(os.path.dirname(HERE), "carrot_functions.py")


def test_mode1_never_launches_a_stopped_car():
  assert tlv_start_xstate_ok(XState.e2eStop, 1) is True
  assert tlv_start_xstate_ok(XState.e2eStopped, 1) is False
  assert tlv_start_xstate_ok(XState.e2eStopped, 2) is True
  assert tlv_start_xstate_ok(XState.e2eStopped, 0) is True
  assert tlv_start_xstate_ok(XState.e2eCruise, 2) is False


def test_trigger_start_uses_the_mode_gate():
  with open(SOURCE, encoding="utf-8") as source:
    text = source.read()
  assert text.count("tlv_start_xstate_ok(self.xState, self.trafficLightDetectMode)") == 1
