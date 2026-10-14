# Bundled traffic-light models (fp32)

Bundled fp32 models: the r8s50a detector `tl_detector_r8s50a.onnx`
(sha256 5f108f62567557e0ac35902a444812230b3da4bd9f0729163a8ff4c3bb909f94) and the
r9 lamp `tl_lamp_r9.onnx` (sha256 665d3649c0392a6d306a822dfe905070e1149cea68f3e6c5b0cddfbd3dc38be7).
r8s50a: single-class (any traffic-light lamp), 6 raw heads; colour comes only from the r9 lamp classifier.
r9: trained on device-path crops (crops.py LampCropBuffer, 25 % pad, NV12) with ±25 % box jitter;
8-class lamp output (CONTRACT classes).
Trained on AI Hub traffic-light data for this device feature; not a general-purpose model.
