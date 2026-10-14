# Bundled traffic-light models (fp32)

Bundled fp32 models: the r8s50a40 detector `tl_detector_r8s50a40.onnx`
(sha256 fc895608b547b0224080e0ee9f183bbfa9a5b441ce5ba381e722f4cc28939162) and the
r9 lamp `tl_lamp_r9.onnx` (sha256 665d3649c0392a6d306a822dfe905070e1149cea68f3e6c5b0cddfbd3dc38be7).
r8s50a40: single-class (any traffic-light lamp), 6 raw heads (r8s50a architecture, 40 epochs; W113); colour comes only from the r9 lamp classifier.
r9: trained on device-path crops (crops.py LampCropBuffer, 25 % pad, NV12) with ±25 % box jitter;
8-class lamp output (CONTRACT classes).
Trained on AI Hub traffic-light data for this device feature; not a general-purpose model.
