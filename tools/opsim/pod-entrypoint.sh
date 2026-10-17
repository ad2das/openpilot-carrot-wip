#!/usr/bin/env bash
# Pod entrypoint for the public opsim image. Runs as PID 1.
# Installs the SSH key RunPod injects via $PUBLIC_KEY, starts sshd, and keeps
# the container alive so the local `opsim-pod` driver can SSH in and run jobs.
set -euo pipefail

# RunPod passes one or more public keys in $PUBLIC_KEY (newline separated).
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  mkdir -p /root/.ssh
  chmod 700 /root/.ssh
  printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  sort -u -o /root/.ssh/authorized_keys /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

mkdir -p /run/sshd
/usr/sbin/sshd -D &

exec sleep infinity
