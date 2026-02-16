# Leonardo Deployment Spec (Agent-Executable)

This document is for automated deployment agents. It is not a narrative guide.

- Target topology: Raspberry Pi 4 client + Ubuntu Mini PC server
- Transport: LAN `ws://` (no TLS termination in this spec)
- Service manager: `systemd`
- Host OS assumption: fresh Ubuntu-class install with `sudo` access

## 0. Required Inputs

Set these variables before running any steps.

```bash
export SERVER_HOSTNAME="minipc"
export SERVER_IP="192.168.1.50"
export SERVER_USER="leonardo"
export CLIENT_HOSTNAME="rpi4"
export CLIENT_IP="192.168.1.60"
export CLIENT_USER="leonardo-client"
export APP_DIR="/opt/leonardo_v1"
export REPO_URL="https://github.com/andi-zehan/voice-assistent.git"
export REPO_REF="main"
export LEO_OPENROUTER_API_KEY="REPLACE_ME"
export SERVER_PORT="8765"
export PIPER_EN_VOICE="en_GB-jenny_dioco-medium"
export PIPER_DE_VOICE="de_DE-thorsten-medium"
export PIPER_EN_ONNX_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx"
export PIPER_EN_JSON_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json"
export PIPER_DE_ONNX_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx"
export PIPER_DE_JSON_URL="https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json"
```

Hard fail if any required variable is missing:

```bash
set -euo pipefail
for v in SERVER_HOSTNAME SERVER_IP SERVER_USER CLIENT_HOSTNAME CLIENT_IP CLIENT_USER APP_DIR REPO_URL REPO_REF LEO_OPENROUTER_API_KEY SERVER_PORT PIPER_EN_VOICE PIPER_DE_VOICE PIPER_EN_ONNX_URL PIPER_EN_JSON_URL PIPER_DE_ONNX_URL PIPER_DE_JSON_URL; do
  test -n "${!v:-}" || { echo "Missing env var: $v" >&2; exit 1; }
done
```

Expected result: no output, exit code `0`.

## 0.1 Script-First Execution (Preferred For Agents)

The repository includes executable deployment scripts:

- `scripts/deploy_server.sh`
- `scripts/deploy_client.sh`

Run them on their respective hosts after exporting inputs:

```bash
# On server host:
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-/tmp/leonardo_bootstrap_server}"
rm -rf "$BOOTSTRAP_DIR"
git clone "$REPO_URL" "$BOOTSTRAP_DIR"
cd "$BOOTSTRAP_DIR"
./scripts/deploy_server.sh

# On client host:
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-/tmp/leonardo_bootstrap_client}"
rm -rf "$BOOTSTRAP_DIR"
git clone "$REPO_URL" "$BOOTSTRAP_DIR"
cd "$BOOTSTRAP_DIR"
./scripts/deploy_client.sh
```

Expected result:
- scripts finish with exit code `0`
- services are enabled and active

Note:
- manual sections below remain the source-of-truth breakdown of each script action.

## 1. Preflight (Run On Each Host)

```bash
set -euo pipefail
sudo -n true
. /etc/os-release
echo "Host: $(hostname) | OS: ${PRETTY_NAME}"
python3 --version || true
```

Expected result:
- `sudo -n true` succeeds
- host info printed
- Python version printed (if missing, it will be installed later)

Failure handling:
- if `sudo -n true` fails, grant passwordless sudo to the deployment agent identity or run in privileged session.

## 2. Provision Server (Run On Ubuntu Mini PC)

### 2.1 Install prerequisites

```bash
set -euo pipefail
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl ca-certificates build-essential \
  python3 python3-venv python3-pip \
  ffmpeg libsndfile1 netcat-openbsd
```

Expected result: all packages install with exit code `0`.

### 2.2 Create service user and app directory

```bash
set -euo pipefail
if ! id -u "$SERVER_USER" >/dev/null 2>&1; then
  sudo useradd --system --create-home --home "/var/lib/${SERVER_USER}" --shell /usr/sbin/nologin "$SERVER_USER"
fi
sudo mkdir -p "$APP_DIR"
sudo chown -R "$SERVER_USER:$SERVER_USER" "$APP_DIR"
```

Expected result: `id -u "$SERVER_USER"` succeeds and `APP_DIR` is writable by `SERVER_USER`.

### 2.3 Clone/update repository at exact ref

```bash
set -euo pipefail
if [ ! -d "$APP_DIR/.git" ]; then
  sudo -u "$SERVER_USER" git clone "$REPO_URL" "$APP_DIR"
fi
sudo -u "$SERVER_USER" git -C "$APP_DIR" remote set-url origin "$REPO_URL"
sudo -u "$SERVER_USER" git -C "$APP_DIR" fetch --all --tags --prune
sudo -u "$SERVER_USER" git -C "$APP_DIR" checkout --force "$REPO_REF"
if sudo -u "$SERVER_USER" git -C "$APP_DIR" rev-parse --verify --quiet "origin/$REPO_REF" >/dev/null; then
  sudo -u "$SERVER_USER" git -C "$APP_DIR" reset --hard "origin/$REPO_REF"
fi
```

Expected result: `git -C "$APP_DIR" rev-parse --short HEAD` returns a commit hash.

### 2.4 Create venv and install server dependencies

```bash
set -euo pipefail
sudo -u "$SERVER_USER" python3 -m venv "$APP_DIR/.venv"
sudo -u "$SERVER_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip setuptools wheel
if ! sudo -u "$SERVER_USER" "$APP_DIR/.venv/bin/pip" install -e "${APP_DIR}[server]"; then
  sudo -u "$SERVER_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/server_requirements.txt"
fi
```

Expected result: install exits `0`.

### 2.5 Download Piper voice files

```bash
set -euo pipefail
sudo -u "$SERVER_USER" mkdir -p "$APP_DIR/models/piper"
sudo -u "$SERVER_USER" curl -fsSL "$PIPER_EN_ONNX_URL" -o "$APP_DIR/models/piper/${PIPER_EN_VOICE}.onnx"
sudo -u "$SERVER_USER" curl -fsSL "$PIPER_EN_JSON_URL" -o "$APP_DIR/models/piper/${PIPER_EN_VOICE}.onnx.json"
sudo -u "$SERVER_USER" curl -fsSL "$PIPER_DE_ONNX_URL" -o "$APP_DIR/models/piper/${PIPER_DE_VOICE}.onnx"
sudo -u "$SERVER_USER" curl -fsSL "$PIPER_DE_JSON_URL" -o "$APP_DIR/models/piper/${PIPER_DE_VOICE}.onnx.json"
```

Verification:

```bash
set -euo pipefail
test -s "$APP_DIR/models/piper/${PIPER_EN_VOICE}.onnx"
test -s "$APP_DIR/models/piper/${PIPER_EN_VOICE}.onnx.json"
test -s "$APP_DIR/models/piper/${PIPER_DE_VOICE}.onnx"
test -s "$APP_DIR/models/piper/${PIPER_DE_VOICE}.onnx.json"
```

### 2.6 Configure server YAML non-interactively

```bash
set -euo pipefail
sudo cp -an "$APP_DIR/server/config.yaml" "$APP_DIR/server/config.yaml.bak.predeploy" || true
sudo -u "$SERVER_USER" APP_DIR="$APP_DIR" SERVER_PORT="$SERVER_PORT" PIPER_EN_VOICE="$PIPER_EN_VOICE" PIPER_DE_VOICE="$PIPER_DE_VOICE" "$APP_DIR/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
import yaml

cfg_path = Path(os.environ["APP_DIR"]) / "server/config.yaml"
cfg = yaml.safe_load(cfg_path.read_text())

cfg.setdefault("server", {})
cfg["server"]["host"] = "0.0.0.0"
cfg["server"]["port"] = int(os.environ["SERVER_PORT"])

cfg.setdefault("tts", {})
cfg["tts"]["engine"] = "piper"
cfg["tts"]["model_dir"] = "models/piper"
cfg["tts"]["default_language"] = "en"
cfg["tts"]["voices"] = {
    "en": {"piper_voice": os.environ["PIPER_EN_VOICE"]},
    "de": {"piper_voice": os.environ["PIPER_DE_VOICE"]},
}

cfg.setdefault("protocol", {})
cfg["protocol"]["audio_mismatch_reject_ratio"] = 0.2

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
```

Expected result: `server/config.yaml` exists and is valid YAML.

### 2.7 Write secure environment file (API key)

```bash
set -euo pipefail
sudo install -d -m 0750 -o root -g "$SERVER_USER" /etc/leonardo
sudo LEO_OPENROUTER_API_KEY="$LEO_OPENROUTER_API_KEY" bash -c 'umask 027; printf "LEO_OPENROUTER_API_KEY=%s\n" "$LEO_OPENROUTER_API_KEY" > /etc/leonardo/server.env'
sudo chown root:"$SERVER_USER" /etc/leonardo/server.env
sudo chmod 0640 /etc/leonardo/server.env
```

Expected result: `/etc/leonardo/server.env` readable by group `SERVER_USER`, not world-readable.

### 2.8 Create and start server `systemd` service

```bash
set -euo pipefail
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
```

### 2.9 Server validation gates

```bash
set -euo pipefail
sudo systemctl is-active --quiet leonardo-server.service
sudo systemctl is-enabled --quiet leonardo-server.service
ss -ltn "( sport = :${SERVER_PORT} )" | grep -q ":${SERVER_PORT}"
sudo journalctl -u leonardo-server.service -n 80 --no-pager | tail -n 40
```

Expected result:
- service active + enabled
- listener on `${SERVER_PORT}`
- no crash loop messages in logs

Optional firewall (if `ufw` is used on server):

```bash
set -euo pipefail
if command -v ufw >/dev/null 2>&1; then
  sudo ufw allow "${SERVER_PORT}/tcp"
fi
```

## 3. Provision Client (Run On Raspberry Pi 4)

### 3.1 Install prerequisites

```bash
set -euo pipefail
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git curl ca-certificates build-essential \
  python3 python3-venv python3-pip \
  portaudio19-dev libportaudio2 libsndfile1 ffmpeg netcat-openbsd
```

Expected result: package install exits `0`.

### 3.2 Create service user and app directory

```bash
set -euo pipefail
if ! id -u "$CLIENT_USER" >/dev/null 2>&1; then
  sudo useradd --system --create-home --home "/var/lib/${CLIENT_USER}" --shell /usr/sbin/nologin --groups audio "$CLIENT_USER"
fi
sudo usermod -aG audio "$CLIENT_USER"
sudo mkdir -p "$APP_DIR"
sudo chown -R "$CLIENT_USER:$CLIENT_USER" "$APP_DIR"
```

Expected result: `id -u "$CLIENT_USER"` succeeds and user is in group `audio`.

### 3.3 Clone/update repository at exact ref

```bash
set -euo pipefail
if [ ! -d "$APP_DIR/.git" ]; then
  sudo -u "$CLIENT_USER" git clone "$REPO_URL" "$APP_DIR"
fi
sudo -u "$CLIENT_USER" git -C "$APP_DIR" remote set-url origin "$REPO_URL"
sudo -u "$CLIENT_USER" git -C "$APP_DIR" fetch --all --tags --prune
sudo -u "$CLIENT_USER" git -C "$APP_DIR" checkout --force "$REPO_REF"
if sudo -u "$CLIENT_USER" git -C "$APP_DIR" rev-parse --verify --quiet "origin/$REPO_REF" >/dev/null; then
  sudo -u "$CLIENT_USER" git -C "$APP_DIR" reset --hard "origin/$REPO_REF"
fi
```

### 3.4 Create venv and install client dependencies

```bash
set -euo pipefail
sudo -u "$CLIENT_USER" python3 -m venv "$APP_DIR/.venv"
sudo -u "$CLIENT_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip setuptools wheel
if ! sudo -u "$CLIENT_USER" "$APP_DIR/.venv/bin/pip" install -e "${APP_DIR}[client]"; then
  sudo -u "$CLIENT_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/client_requirements.txt"
fi
```

### 3.5 Configure client YAML non-interactively

```bash
set -euo pipefail
sudo cp -an "$APP_DIR/client/config.yaml" "$APP_DIR/client/config.yaml.bak.predeploy" || true
sudo -u "$CLIENT_USER" APP_DIR="$APP_DIR" SERVER_IP="$SERVER_IP" SERVER_PORT="$SERVER_PORT" "$APP_DIR/.venv/bin/python" - <<'PY'
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
```

### 3.6 Client audio preflight

```bash
set -euo pipefail
arecord -l || true
aplay -l || true
sudo -u "$CLIENT_USER" "$APP_DIR/.venv/bin/python" - <<'PY'
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
```

Expected result: at least one input-capable audio device exists.

### 3.7 Create and start client `systemd` service

```bash
set -euo pipefail
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
```

### 3.8 Client validation gates

```bash
set -euo pipefail
sudo systemctl is-active --quiet leonardo-client.service
sudo systemctl is-enabled --quiet leonardo-client.service
nc -z -w3 "$SERVER_IP" "$SERVER_PORT"
sudo journalctl -u leonardo-client.service -n 120 --no-pager | grep -q "Connected to server"
```

Expected result:
- client service active/enabled
- TCP reachability to server
- logs confirm connection established

## 4. Cross-Host Final Gates

Run these checks after both services are up.

### 4.1 Server health

```bash
sudo systemctl is-active --quiet leonardo-server.service
sudo journalctl -u leonardo-server.service -n 120 --no-pager | tail -n 50
```

### 4.2 Client health

```bash
sudo systemctl is-active --quiet leonardo-client.service
sudo journalctl -u leonardo-client.service -n 120 --no-pager | tail -n 50
```

### 4.3 Crash-loop detection (both hosts)

```bash
set -euo pipefail
# Run on server host:
sudo journalctl -u leonardo-server.service -n 200 --no-pager | grep -qi "Start request repeated too quickly" && exit 1 || true
# Run on client host:
sudo journalctl -u leonardo-client.service -n 200 --no-pager | grep -qi "Start request repeated too quickly" && exit 1 || true
```

Expected result: no matches.

## 5. Rollback / Recovery

### 5.1 Stop services

```bash
# Run on client host:
sudo systemctl disable --now leonardo-client.service || true
# Run on server host:
sudo systemctl disable --now leonardo-server.service || true
```

### 5.2 Restore config backups (if present)

```bash
set -euo pipefail
# Run on server host:
[ -f "$APP_DIR/server/config.yaml.bak.predeploy" ] && sudo cp "$APP_DIR/server/config.yaml.bak.predeploy" "$APP_DIR/server/config.yaml" || true
# Run on client host:
[ -f "$APP_DIR/client/config.yaml.bak.predeploy" ] && sudo cp "$APP_DIR/client/config.yaml.bak.predeploy" "$APP_DIR/client/config.yaml" || true
```

### 5.3 Revert code to previous known ref

```bash
set -euo pipefail
# Replace PREVIOUS_REF with your last known good branch/tag/commit.
export PREVIOUS_REF="REPLACE_ME"
# Run on server host:
sudo -u "$SERVER_USER" git -C "$APP_DIR" fetch --all --tags --prune
sudo -u "$SERVER_USER" git -C "$APP_DIR" checkout --force "$PREVIOUS_REF"
# Run on client host:
sudo -u "$CLIENT_USER" git -C "$APP_DIR" fetch --all --tags --prune
sudo -u "$CLIENT_USER" git -C "$APP_DIR" checkout --force "$PREVIOUS_REF"
```

### 5.4 Restart and validate

```bash
# Run on server host:
sudo systemctl enable --now leonardo-server.service
sudo systemctl is-active --quiet leonardo-server.service
# Run on client host:
sudo systemctl enable --now leonardo-client.service
sudo systemctl is-active --quiet leonardo-client.service
```

## 6. Security / Ops Constraints

1. Keep `/etc/leonardo/server.env` mode `0640`, owner `root`, group `SERVER_USER`.
2. Run services as non-root users only.
3. Keep server port restricted to trusted LAN.
4. In shared environments, set these in `server/config.yaml`:
- `metrics.log_transcripts: false`
- `metrics.log_llm_text: false`
