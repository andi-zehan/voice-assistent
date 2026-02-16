#!/usr/bin/env bash
set -euo pipefail

# Agent-executable deployment script for Raspberry Pi 4 client.

CLIENT_HOSTNAME="${CLIENT_HOSTNAME:-rpi4}"
CLIENT_USER="${CLIENT_USER:-leonardo-client}"
APP_DIR="${APP_DIR:-/opt/leonardo_v1}"
REPO_URL="${REPO_URL:-https://github.com/andi-zehan/voice-assistent.git}"
REPO_REF="${REPO_REF:-main}"
SERVER_IP="${SERVER_IP:-192.168.178.66}"
SERVER_PORT="${SERVER_PORT:-8765}"

run_as_client_user() {
  sudo -u "$CLIENT_USER" "$@"
}

echo "[client] starting deployment on host $(hostname)"
sudo -n true

if [[ "$(hostname)" != "$CLIENT_HOSTNAME" ]]; then
  echo "[client] warning: hostname mismatch (expected '$CLIENT_HOSTNAME', got '$(hostname)')"
fi

echo "[client] installing prerequisites"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl ca-certificates build-essential \
  python3 python3-venv python3-pip \
  portaudio19-dev libportaudio2 libsndfile1 ffmpeg netcat-openbsd

echo "[client] creating service user and app dir"
if ! id -u "$CLIENT_USER" >/dev/null 2>&1; then
  sudo useradd --system --create-home --home "/var/lib/${CLIENT_USER}" --shell /usr/sbin/nologin --groups audio "$CLIENT_USER"
fi
sudo usermod -aG audio "$CLIENT_USER"
sudo mkdir -p "$APP_DIR"
sudo chown -R "$CLIENT_USER:$CLIENT_USER" "$APP_DIR"

echo "[client] syncing repo"
if [[ -d "$APP_DIR/.git" ]]; then
  run_as_client_user git -C "$APP_DIR" remote set-url origin "$REPO_URL"
else
  if [[ -n "$(sudo find "$APP_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]]; then
    echo "ERROR: APP_DIR exists and is not an initialized git repository: $APP_DIR" >&2
    exit 1
  fi
  run_as_client_user git clone "$REPO_URL" "$APP_DIR"
fi
run_as_client_user git -C "$APP_DIR" fetch --all --tags --prune
run_as_client_user git -C "$APP_DIR" checkout --force "$REPO_REF"
if run_as_client_user git -C "$APP_DIR" rev-parse --verify --quiet "origin/$REPO_REF" >/dev/null; then
  run_as_client_user git -C "$APP_DIR" reset --hard "origin/$REPO_REF"
fi

echo "[client] creating venv and installing dependencies"
run_as_client_user python3 -m venv "$APP_DIR/.venv"
run_as_client_user "$APP_DIR/.venv/bin/pip" install --upgrade pip setuptools wheel
if ! run_as_client_user "$APP_DIR/.venv/bin/pip" install -e "${APP_DIR}[client]"; then
  run_as_client_user "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/client_requirements.txt"
fi

echo "[client] updating client/config.yaml"
sudo cp -an "$APP_DIR/client/config.yaml" "$APP_DIR/client/config.yaml.bak.predeploy" || true
run_as_client_user env \
  APP_DIR="$APP_DIR" \
  SERVER_IP="$SERVER_IP" \
  SERVER_PORT="$SERVER_PORT" \
  "$APP_DIR/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
import yaml

cfg_path = Path(os.environ["APP_DIR"]) / "client/config.yaml"
cfg = yaml.safe_load(cfg_path.read_text())

cfg.setdefault("server", {})
cfg["server"]["host"] = os.environ["SERVER_IP"]
cfg["server"]["port"] = int(os.environ["SERVER_PORT"])
cfg["server"]["reconnect_min_s"] = 1.0
cfg["server"]["reconnect_max_s"] = 30.0
cfg["server"]["offline_send_buffer_size"] = 200
cfg["server"]["offline_send_ttl_s"] = 5.0

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "[client] audio preflight"
arecord -l || true
aplay -l || true
run_as_client_user "$APP_DIR/.venv/bin/python" - <<'PY'
import sounddevice as sd
devices = sd.query_devices()
print(f"Device count: {len(devices)}")
if not devices:
    raise SystemExit("No audio devices detected")
input_ok = any(d.get("max_input_channels", 0) > 0 for d in devices)
if not input_ok:
    raise SystemExit("No input-capable audio device detected")
print("Audio preflight passed")
PY

echo "[client] installing systemd service"
sudo tee /etc/systemd/system/leonardo-client.service >/dev/null <<EOF
[Unit]
Description=Leonardo Voice Assistant Client
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=${CLIENT_USER}
Group=audio
SupplementaryGroups=audio
WorkingDirectory=${APP_DIR}
ExecStart=${APP_DIR}/.venv/bin/leonardo-client --config ${APP_DIR}/client/config.yaml
Restart=always
RestartSec=2
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=${APP_DIR}

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now leonardo-client.service

echo "[client] validating deployment"
sudo systemctl is-active --quiet leonardo-client.service
sudo systemctl is-enabled --quiet leonardo-client.service
nc -z -w3 "$SERVER_IP" "$SERVER_PORT"

connected=0
for _ in $(seq 1 20); do
  if sudo journalctl -u leonardo-client.service -n 200 --no-pager | grep -q "Connected to server"; then
    connected=1
    break
  fi
  sleep 1
done
if [[ "$connected" -ne 1 ]]; then
  echo "ERROR: client logs do not show 'Connected to server'" >&2
  sudo journalctl -u leonardo-client.service -n 120 --no-pager || true
  exit 1
fi

echo "[client] deployment complete"
sudo systemctl --no-pager --full status leonardo-client.service | sed -n '1,20p'
