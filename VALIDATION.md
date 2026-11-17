# V2 validation record

Source: MrLi002/DSER-QQ @ 983d929512d5af48c12960333438546e04a6daaa.
Environment: Python 3.12.14, torch 2.5.1+cpu, torchvision 0.20.1+cpu,
NumPy 1.26.4, einops 0.8.1, OpenCV 4.11.0.

- 19 unit/integration tests passed.
- Verified that the residual head receives the source feature sampled at a known
  translated E-PCD candidate, rather than the same-coordinate source feature.
- Isolated the new RGB retrieval loss at initialization: Q/K, residual offsets
  and modulation mapping received nonzero gradients while decision and output
  projection were not part of that gradient path. Stage-B offset/m inputs stayed detached.
- Checked that explicit retrieved/base RGB and RGB differences reach the decision head.
- Checked signed modulation priors, invalid-source exclusion, all-invalid and empty
  selection cases, padding, chunk consistency, selection scope and GT-free forward.
- Checked original-to-V2 migration and V1-to-V2 migration safeguards; missing
  unchanged backbone/difficulty keys still fail strict validation.
- Used an actual V1 model checkpoint to initialize V2 training successfully.
- Completed one real DataLoader/training iteration on synthetic BS-ERGB-format
  files, including all losses, backward, clipping, optimizer update, CSV logging
  and atomic checkpoint save. LPIPS used its real operators with a random AlexNet
  backbone solely to avoid downloading pretrained weights in this smoke test.
  Normal training still requests standard pretrained LPIPS.
- Restored the V2 network, optimizer, scheduler and scaler states.
- Passed 256x256 inference and inference-wrapper padding/unpadding for 65x97.
- Original BASR mode matched the source network output tuple exactly: maximum
  absolute tensor difference 0.0.

This is functional validation. No full real-data training, quality improvement,
CUDA/AMP/multi-GPU validation, GPU memory measurement or speed claim is made.
