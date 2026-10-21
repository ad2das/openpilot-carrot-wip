# Bundled traffic-light models (fp32)

Bundled fp32 models: the r8s50a40 detector `tl_detector_r8s50a40.onnx`
(sha256 fc895608b547b0224080e0ee9f183bbfa9a5b441ce5ba381e722f4cc28939162) and the
r9 lamp `tl_lamp_r9.onnx` (sha256 665d3649c0392a6d306a822dfe905070e1149cea68f3e6c5b0cddfbd3dc38be7).
r8s50a40: single-class (any traffic-light lamp), 6 raw heads (r8s50a architecture, 40 epochs; W113); colour comes only from the r9 lamp classifier.
r9: trained on device-path crops (crops.py LampCropBuffer, 25 % pad, NV12) with ±25 % box jitter;
8-class lamp output (CONTRACT classes).
Trained on AI Hub traffic-light data for this device feature; not a general-purpose model.

## DSP detector DLCs (for the SNPE DSP backend)

Checked on the C3X by the one-shot probe snpe3 (2026-10-21, since removed): on the Hexagon DSP both
run at ~16 ms median per frame and match the SNPE CPU float run on 35 real frames (23/23 dets).
The onroad TLV pipeline does not load them yet. Both take `img` NHWC `[1,320,640,3]` float 0..255 and
return one `heads` output `[1,273000]` that `tl_detector_r8s50a40_heads.json` splits into the six heads.

| file | source | quantizer | sha256 |
| --- | --- | --- | --- |
| `tl_detector_r8s50a40_f32.dlc` | `data/wsnpe6/det_f32.dlc` (float, 2,360,359 B) | SNPE 1.68 converter, float (unquantized) | 04b1d6845a40ed62fe393c2fdc363b999ee2739b1ee78d302b986679ad7ef25a |
| `tl_detector_r8s50a40_q.dlc` | `data/wsnpe6/det_q140a.dlc` (int8, 764,552 B) | SNPE 1.40.0.2130 | 996b7086bc27f274cadab97362e0632e2d104749ff66fa6b7eaad458d9b37367 |
| `tl_detector_r8s50a40_heads.json` | `data/wsnpe6/heads.json` (273000-float head split) | — | c3f82afb9917db156fd36967f63d062a740d89c5a07cfcd7582b32283c8e3966 |
