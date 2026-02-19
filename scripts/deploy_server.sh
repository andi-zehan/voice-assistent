#!/usr/bin/env bash
set -euo pipefail

# Agent-executable deployment script for Ubuntu Mini PC server.

SERVER_HOSTNAME="${SERVER_HOSTNAME:-mediabox}"
SERVER_USER="${SERVER_USER:-leonardo}"
APP_DIR="${APP_DIR:-/opt/leonardo_v1}"
REPO_URL="${REPO_URL:-https://github.com/andi-zehan/voice-assistent.git}"
REPO_REF="${REPO_REF:-main}"
LEO_OPENROUTER_API_KEY="${LEO_OPENROUTER_API_KEY:-}"
SERVER_PORT="${SERVER_PORT:-8765}"
PIPER_EN_VOICE="${PIPER_EN_VOICE:-en_GB-southern_english_female-low}"
PIPER_DE_VOICE="${PIPER_DE_VOICE:-de_DE-thorsten-medium}"
PIPER_EN_ONNX_URL="${PIPER_EN_ONNX_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/southern_english_female/low/en_GB-southern_english_female-low.onnx}"
PIPER_EN_JSON_URL="${PIPER_EN_JSON_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/southern_english_female/low/en_GB-southern_english_female-low.onnx.json}"
PIPER_DE_ONNX_URL="${PIPER_DE_ONNX_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx}"
PIPER_DE_JSON_URL="${PIPER_DE_JSON_URL:-https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json}"
TTS_ENGINE="${TTS_ENGINE:-piper}"
KOKORO_EN_VOICE="${KOKORO_EN_VOICE:-af_bella}"
ENABLE_UFW_RULE="${ENABLE_UFW_RULE:-false}"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: Missing required env var: $name" >&2
    exit 1
  fi
}

run_as_server_user() {
  sudo -u "$SERVER_USER" "$@"
}

echo "[server] starting deployment on host $(hostname)"
sudo -n true

require_env LEO_OPENROUTER_API_KEY

if [[ "$(hostname)" != "$SERVER_HOSTNAME" ]]; then
  echo "[server] warning: hostname mismatch (expected '$SERVER_HOSTNAME', got '$(hostname)')"
fi

echo "[server] installing prerequisites"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y 2>/dev/null || \
  echo "[server] warning: apt-get update had errors (non-fatal, continuing)"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl ca-certificates build-essential \
  python3 python3-venv python3-pip \
  ffmpeg libsndfile1 netcat-openbsd \
  espeak-ng

echo "[server] creating service user and app dir"
if ! id -u "$SERVER_USER" >/dev/null 2>&1; then
  sudo useradd --system --create-home --home "/var/lib/${SERVER_USER}" --shell /usr/sbin/nologin "$SERVER_USER"
fi
sudo mkdir -p "$APP_DIR"
sudo chown -R "$SERVER_USER:$SERVER_USER" "$APP_DIR"

echo "[server] syncing repo"
if [[ -d "$APP_DIR/.git" ]]; then
  run_as_server_user git -C "$APP_DIR" remote set-url origin "$REPO_URL"
else
  if [[ -n "$(sudo find "$APP_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null || true)" ]]; then
    echo "ERROR: APP_DIR exists and is not an initialized git repository: $APP_DIR" >&2
    exit 1
  fi
  run_as_server_user git clone "$REPO_URL" "$APP_DIR"
fi
run_as_server_user git -C "$APP_DIR" fetch --all --tags --prune
run_as_server_user git -C "$APP_DIR" checkout --force "$REPO_REF"
if run_as_server_user git -C "$APP_DIR" rev-parse --verify --quiet "origin/$REPO_REF" >/dev/null; then
  run_as_server_user git -C "$APP_DIR" reset --hard "origin/$REPO_REF"
fi

echo "[server] creating venv and installing dependencies"
run_as_server_user python3 -m venv "$APP_DIR/.venv"
run_as_server_user "$APP_DIR/.venv/bin/pip" install --upgrade pip setuptools wheel
if ! run_as_server_user "$APP_DIR/.venv/bin/pip" install -e "${APP_DIR}[server]"; then
  run_as_server_user "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/server_requirements.txt"
fi

echo "[server] downloading piper voices"
run_as_server_user mkdir -p "$APP_DIR/models/piper"
run_as_server_user curl -fL --retry 3 --retry-delay 1 "$PIPER_EN_ONNX_URL" -o "$APP_DIR/models/piper/${PIPER_EN_VOICE}.onnx"
run_as_server_user curl -fL --retry 3 --retry-delay 1 "$PIPER_EN_JSON_URL" -o "$APP_DIR/models/piper/${PIPER_EN_VOICE}.onnx.json"
run_as_server_user curl -fL --retry 3 --retry-delay 1 "$PIPER_DE_ONNX_URL" -o "$APP_DIR/models/piper/${PIPER_DE_VOICE}.onnx"
run_as_server_user curl -fL --retry 3 --retry-delay 1 "$PIPER_DE_JSON_URL" -o "$APP_DIR/models/piper/${PIPER_DE_VOICE}.onnx.json"

echo "[server] updating server/config.yaml"
sudo cp -an "$APP_DIR/server/config.yaml" "$APP_DIR/server/config.yaml.bak.predeploy" || true
run_as_server_user env \
  APP_DIR="$APP_DIR" \
  SERVER_PORT="$SERVER_PORT" \
  TTS_ENGINE="$TTS_ENGINE" \
  KOKORO_EN_VOICE="$KOKORO_EN_VOICE" \
  PIPER_EN_VOICE="$PIPER_EN_VOICE" \
  PIPER_DE_VOICE="$PIPER_DE_VOICE" \
  "$APP_DIR/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
import yaml

cfg_path = Path(os.environ["APP_DIR"]) / "server/config.yaml"
cfg = yaml.safe_load(cfg_path.read_text())

cfg.setdefault("server", {})
cfg["server"]["host"] = "0.0.0.0"
cfg["server"]["port"] = int(os.environ["SERVER_PORT"])

cfg.setdefault("tts", {})
cfg["tts"]["engine"] = os.environ.get("TTS_ENGINE", "piper")
cfg["tts"]["model_dir"] = "models/piper"
cfg["tts"]["default_language"] = "en"
if os.environ.get("TTS_ENGINE") == "kokoro":
    cfg["tts"]["speed"] = 1.0
    cfg["tts"]["voices"] = {
        "en": {"kokoro_voice": os.environ["KOKORO_EN_VOICE"]},
        "de": {"piper_voice": os.environ["PIPER_DE_VOICE"]},
    }
else:
    cfg["tts"]["voices"] = {
        "en": {"piper_voice": os.environ["PIPER_EN_VOICE"]},
        "de": {"piper_voice": os.environ["PIPER_DE_VOICE"]},
    }

cfg.setdefault("protocol", {})
cfg["protocol"]["audio_mismatch_reject_ratio"] = 0.2

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY

echo "[server] writing secure env file"
sudo install -d -m 0750 -o root -g "$SERVER_USER" /etc/leonardo
sudo env LEO_OPENROUTER_API_KEY="$LEO_OPENROUTER_API_KEY" bash -c 'umask 027; printf "LEO_OPENROUTER_API_KEY=%s\n" "$LEO_OPENROUTER_API_KEY" > /etc/leonardo/server.env'
sudo chown root:"$SERVER_USER" /etc/leonardo/server.env
sudo chmod 0640 /etc/leonardo/server.env

echo "[server] installing systemd service"
sudo tee /etc/systemd/system/leonardo-server.service >/dev/null <<EOF
[Unit]
Description=Leonardo Voice Assistant Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVER_USER}
Group=${SERVER_USER}
WorkingDirectory=${APP_DIR}
EnvironmentFile=/etc/leonardo/server.env
ExecStart=${APP_DIR}/.venv/bin/leonardo-server --config ${APP_DIR}/server/config.yaml
Restart=always
RestartSec=3
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true
ReadWritePaths=${APP_DIR}

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now leonardo-server.service

if [[ "$ENABLE_UFW_RULE" == "true" ]] && command -v ufw >/dev/null 2>&1; then
  echo "[server] applying ufw rule for port ${SERVER_PORT}"
  sudo ufw allow "${SERVER_PORT}/tcp"
fi

echo "[server] validating deployment"
sudo systemctl is-active --quiet leonardo-server.service
sudo systemctl is-enabled --quiet leonardo-server.service
ss -ltn "( sport = :${SERVER_PORT} )" | grep -q ":${SERVER_PORT}"
for f in \
  "$APP_DIR/models/piper/${PIPER_EN_VOICE}.onnx" \
  "$APP_DIR/models/piper/${PIPER_EN_VOICE}.onnx.json" \
  "$APP_DIR/models/piper/${PIPER_DE_VOICE}.onnx" \
  "$APP_DIR/models/piper/${PIPER_DE_VOICE}.onnx.json"; do
  test -s "$f"
done

echo "[server] deployment complete"
sudo systemctl --no-pager --full status leonardo-server.service | sed -n '1,20p'
