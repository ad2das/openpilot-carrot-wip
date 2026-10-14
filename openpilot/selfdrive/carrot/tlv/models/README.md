# Bundled traffic-light models (fp32)

Bundled fp32 models: the R3b detector `tl_detector_r3b.onnx`
(sha256 d106b3f9f9d100ecdd71bd743683eac21f7b06b8e5ff2abd96841e934fb8db09) and the
r9 lamp `tl_lamp_r9.onnx` (sha256 665d3649c0392a6d306a822dfe905070e1149cea68f3e6c5b0cddfbd3dc38be7).
r9: trained on device-path crops (crops.py LampCropBuffer, 25 % pad, NV12) with ±25 % box jitter;
same 8-class I/O contract as r3b.
Trained on AI Hub traffic-light data for this device feature; not a general-purpose model.
