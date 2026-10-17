# Public CPU-only opsim image: Ubuntu 24.04 + openpilot build deps + uv venv + a
# built carrot-wip checkout. A fresh RunPod CPU pod pulls this, then runs an
# incremental `git fetch <ref> && scons` for the ref under test.
#
# Contains public repo code and public packages only: no datasets, keys, logs or
# private paths. The pod is the only thing that talks to the pr-fork remote.
#
# Build context is the checked-out ad2das/openpilot-carrot-wip tree (see
# .github/workflows/opsim-image.yml), including .git so the pod can fetch.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_VERSION=0.13.1 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    PYTHONUNBUFFERED=1 \
    OPENPILOT=/root/openpilot

# System packages: mirrors tools/setup_dependencies.sh (linux deps + the
# PlotJuggler/Qt tool deps so the same setup path is reproducible) plus the
# extras the image needs (python venv, ssh, scons prerequisites).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates build-essential curl libcurl4-openssl-dev locales git \
      python3 python3-dev python3-venv python3-pip \
      libzmq3-dev liblz4-dev libzstd-dev \
      libgles-dev libqt5charts5-dev libqt5svg5-dev libqt5websockets5-dev \
      libqt5serialport5-dev libqt5opengl5-dev libqt5x11extras5-dev \
      qtbase5-dev qtwayland5 libprotoc-dev \
      ffmpeg libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
      libsdl2-dev libssl-dev pkg-config \
      openssh-server tmux zstd jq sudo \
    && rm -rf /var/lib/apt/lists/* \
    && locale-gen en_US.UTF-8 || true

# uv (matches the local WSL toolchain version).
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh \
    && uv --version

# Copy the built-context checkout (includes .git for later incremental fetch).
COPY . ${OPENPILOT}
WORKDIR ${OPENPILOT}

# System + python deps, then a full scons build so the checkout ships prebuilt.
RUN ln -sf /usr/bin/python3 /usr/local/bin/python && \
    uv sync --frozen --all-extras && \
    .venv/bin/python -c "import numpy, capnp; print('venv ok')" && \
    scons -j"$(nproc)" || scons

# SSH for the pod; root login with injected keys (RunPod supplies PUBLIC_KEY).
RUN mkdir -p /var/run/sshd /root/.ssh && chmod 700 /root/.ssh && \
    sed -i 's/#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -i 's/#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

# Entrypoint: install RunPod's injected PUBLIC_KEY, start sshd, keep the pod alive.
COPY tools/opsim/pod-entrypoint.sh /usr/local/bin/pod-entrypoint.sh
RUN chmod +x /usr/local/bin/pod-entrypoint.sh

EXPOSE 22
CMD ["/usr/local/bin/pod-entrypoint.sh"]
