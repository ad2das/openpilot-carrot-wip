# Public GPU training image for the RunPod "trainpod" driver: python 3.11 +
# torch cu126 + the exact carrot-tl-vision training pins.
#
# Environment only: public packages, no datasets, no repo code, no keys. Code
# and data are streamed to a fresh pod per job by bin/trainpod and deleted with
# the pod (no network volume, nothing left on RunPod after a job).
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      openssh-server zstd tar procps tmux ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY tools/tltrain/requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cu126 \
      torch==2.14.1 torchvision==0.29.1 && \
    pip install -r /tmp/requirements.txt && \
    python -c "import torch, torchvision, ultralytics, albumentations, onnx, onnxruntime, cv2, numpy, scipy; print('env ok', torch.__version__)" && \
    pip list | grep -i opencv

# SSH for the pod; root login with injected keys (RunPod supplies PUBLIC_KEY).
RUN mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh && \
    sed -i 's/#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    rm -f /etc/ssh/ssh_host_*

COPY tools/tltrain/pod-entrypoint.sh /usr/local/bin/pod-entrypoint.sh
RUN chmod +x /usr/local/bin/pod-entrypoint.sh

EXPOSE 22
CMD ["/usr/local/bin/pod-entrypoint.sh"]
