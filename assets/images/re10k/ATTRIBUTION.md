# Attribution

Source: YoNoSplat project page demo video (`re10k_0f12b97e0e4c7e21.mp4`)
- Page: https://botaoye.github.io/yonosplat/
- Video: https://botaoye.github.io/yonosplat/assets/video/re10k_0f12b97e0e4c7e21.mp4

This fixture provides three related views from one RE10K-style demo scene for
multi-view reconstruction testing.

Extraction (reproducible):
- Tool: `ffmpeg 7.0.2 static`
- Timestamps: `0.4s`, `3.2s`, `6.4s`
- Command pattern:
  - `ffmpeg -ss <time> -i re10k_0f12b97e0e4c7e21.mp4 -frames:v 1 view_0N.png`

Notes:
- Images are resized/cropped by the pipeline to model input size at runtime.
- Keep this attribution alongside the fixture images.
