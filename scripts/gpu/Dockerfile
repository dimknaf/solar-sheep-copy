# NVIDIA's pre-built Isaac Lab 3.0 EA container plus the video extra it leaves out.
# Without it, `isaaclab play/train --video` fails: "Video recording with `--video` requires
# MoviePy and its imageio-ffmpeg backend, which are not installed by default" - the fix that
# message prescribes for this (non-uv) environment is exactly this pip install.
#   docker build -t solar/isaac-lab:3.0.0-rc1-video scripts/gpu
FROM nvcr.io/nvidia/isaac-lab:3.0.0-rc1
USER root
RUN /isaac-sim/python.sh -m pip install --no-cache-dir "moviepy>=1.0.3,<2.0.0.dev0" imageio-ffmpeg
USER isaaclab
