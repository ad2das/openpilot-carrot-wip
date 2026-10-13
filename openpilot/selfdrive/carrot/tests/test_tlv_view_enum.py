from openpilot.cereal import messaging
from openpilot.selfdrive.carrot.tlv import tlvd


def test_every_pipeline_view_publishes():
  for view in ("far", "near"):
    msg = messaging.new_message("carrotTlv")
    msg.carrotTlv.view = tlvd.CAPNP_VIEW[view]
    assert str(msg.carrotTlv.view) == tlvd.CAPNP_VIEW[view]
