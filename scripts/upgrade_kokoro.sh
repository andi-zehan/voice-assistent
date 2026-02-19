#!/usr/bin/env bash
set -euo pipefail

# ─── Upgrade Leonardo server to Kokoro TTS ───────────────────────
# Run on the server with: sudo bash upgrade_kokoro.sh
#
# What this does:
#   1. Stops the leonardo-server service
#   2. Pulls latest code from feat/client-server-split
#   3. Installs espeak-ng (Kokoro phonemizer dependency)
#   4. Installs kokoro pip package
#   5. Updates server/config.yaml to use Kokoro for English, Piper for German
#   6. Restarts the service and tails the logs
#
# Prerequisites:
#   - Existing Leonardo server deployment at /opt/leonardo_v1
#   - Run as root or with passwordless sudo

APP_DIR="${APP_DIR:-/opt/leonardo_v1}"
SERVER_USER="${SERVER_USER:-leonardo}"
REPO_REF="${REPO_REF:-feat/client-server-split}"
KOKORO_EN_VOICE="${KOKORO_EN_VOICE:-af_bella}"
PIPER_DE_VOICE="${PIPER_DE_VOICE:-de_DE-thorsten-medium}"

run_as() { sudo -u "$SERVER_USER" "$@"; }

echo "═══════════════════════════════════════════════════"
echo "  Leonardo TTS Upgrade: Piper → Kokoro-82M"
echo "═══════════════════════════════════════════════════"

# ── Preflight checks ─────────────────────────────────────────────
if [[ ! -d "$APP_DIR/.git" ]]; then
  echo "ERROR: $APP_DIR is not a git repository. Run deploy_server.sh first." >&2
  exit 1
fi
if [[ ! -f "$APP_DIR/.venv/bin/pip" ]]; then
  echo "ERROR: No virtualenv found at $APP_DIR/.venv" >&2
  exit 1
fi

# ── Step 1: Stop service ─────────────────────────────────────────
echo ""
echo "[1/6] Stopping leonardo-server..."
if sudo systemctl is-active --quiet leonardo-server.service 2>/dev/null; then
  sudo systemctl stop leonardo-server.service
  echo "      Service stopped."
else
  echo "      Service was not running."
fi

# ── Step 2: Pull latest code ─────────────────────────────────────
echo ""
echo "[2/6] Pulling latest code (branch: $REPO_REF)..."
run_as git -C "$APP_DIR" fetch --all --prune
run_as git -C "$APP_DIR" checkout --force "$REPO_REF"
if run_as git -C "$APP_DIR" rev-parse --verify --quiet "origin/$REPO_REF" >/dev/null; then
  run_as git -C "$APP_DIR" reset --hard "origin/$REPO_REF"
fi
echo "      HEAD: $(run_as git -C "$APP_DIR" log --oneline -1)"

# ── Step 3: Install espeak-ng ────────────────────────────────────
echo ""
echo "[3/6] Installing espeak-ng (Kokoro phonemizer)..."
if command -v espeak-ng >/dev/null 2>&1; then
  echo "      Already installed: $(espeak-ng --version 2>&1 | head -1)"
else
  # apt-get update may fail due to unrelated broken repos — try install anyway
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -y -qq 2>/dev/null || \
    echo "      Warning: apt-get update had errors (non-fatal, continuing)"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq espeak-ng
  echo "      Installed."
fi

# ── Step 4: Install Python dependencies ──────────────────────────
echo ""
echo "[4/6] Installing kokoro and updating dependencies..."
run_as "$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip setuptools wheel
if ! run_as "$APP_DIR/.venv/bin/pip" install --quiet -e "${APP_DIR}[server]" 2>/dev/null; then
  run_as "$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/server_requirements.txt"
fi
# Verify kokoro is importable
run_as "$APP_DIR/.venv/bin/python" -c "import kokoro; print(f'      kokoro {kokoro.__version__} installed')" 2>/dev/null \
  || run_as "$APP_DIR/.venv/bin/python" -c "import kokoro; print('      kokoro installed (version unknown)')"

# ── Step 5: Update config.yaml ───────────────────────────────────
echo ""
echo "[5/6] Updating config.yaml (engine=kokoro, voice=$KOKORO_EN_VOICE)..."
sudo cp -a "$APP_DIR/server/config.yaml" "$APP_DIR/server/config.yaml.bak.$(date +%Y%m%d%H%M%S)"
run_as env \
  APP_DIR="$APP_DIR" \
  KOKORO_EN_VOICE="$KOKORO_EN_VOICE" \
  PIPER_DE_VOICE="$PIPER_DE_VOICE" \
  "$APP_DIR/.venv/bin/python" - <<'PYTHON'
import os
from pathlib import Path
import yaml

cfg_path = Path(os.environ["APP_DIR"]) / "server/config.yaml"
cfg = yaml.safe_load(cfg_path.read_text())

cfg.setdefault("tts", {})
cfg["tts"]["engine"] = "kokoro"
cfg["tts"]["speed"] = 1.0
cfg["tts"]["voices"] = {
    "en": {"kokoro_voice": os.environ["KOKORO_EN_VOICE"]},
    "de": {"piper_voice": os.environ["PIPER_DE_VOICE"]},
}

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"      Engine: kokoro")
print(f"      EN voice: {os.environ['KOKORO_EN_VOICE']}")
print(f"      DE voice: {os.environ['PIPER_DE_VOICE']} (Piper fallback)")
PYTHON

# ── Step 6: Restart and verify ───────────────────────────────────
echo ""
echo "[6/6] Starting leonardo-server..."
sudo systemctl start leonardo-server.service

# Wait briefly for service to start
sleep 3

if sudo systemctl is-active --quiet leonardo-server.service; then
  echo "      Service is RUNNING."
else
  echo "      WARNING: Service failed to start!" >&2
  sudo journalctl -u leonardo-server.service -n 20 --no-pager >&2
  exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Upgrade complete! Tailing logs (Ctrl-C to stop)"
echo "═══════════════════════════════════════════════════"
echo ""
sudo journalctl -u leonardo-server.service -n 30 --no-pager -f
