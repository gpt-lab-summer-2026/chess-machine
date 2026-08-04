#!/usr/bin/env bash
# Autostart the chess machine's voice/game loop on the console it's plugged into.
#
# Called from ~/.bash_profile on the autologin tty (see docs/SETUP_PI.md #7), so
# the "you> ..." / "[speaker] ..." console mirror lands on the physical screen
# attached to the Pi, not a headless log file only.
#
# llama-server runs as its OWN systemd service (deploy/llama-server.service) --
# not started here -- so it survives independently of this console session and
# systemd supervises its restarts. This script just waits for it to answer
# before the first move, then runs chessmachine and restarts it if it crashes
# (a serial glitch, a killed llama-server, ...). Ctrl-C breaks the restart loop
# and hands back a normal shell instead of relaunching under you.
set -u

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$REPO_DIR/.venv/bin/python"
LOG_DIR="$REPO_DIR/logs"
LOG_FILE="$LOG_DIR/chessmachine.log"
SLM_HEALTH_URL="http://127.0.0.1:8080/health"
SLM_WAIT_S=60
RESTART_DELAY_S=3

mkdir -p "$LOG_DIR"
cd "$REPO_DIR" || exit 1

if [ ! -x "$PYTHON" ]; then
    echo "boot_chessmachine.sh: no venv at $PYTHON -- run the setup in docs/SETUP_PI.md first." >&2
    exit 1
fi

# chessmachine's own SLM warmup already tolerates a slow/absent server (it falls
# back to rule-based), so this wait is only to avoid the FIRST move racing a cold
# prefill -- not a hard requirement, hence it gives up and starts anyway.
echo "waiting up to ${SLM_WAIT_S}s for llama-server..."
waited=0
while ! curl -fsS -m 2 "$SLM_HEALTH_URL" >/dev/null 2>&1; do
    waited=$((waited + 2))
    if [ "$waited" -ge "$SLM_WAIT_S" ]; then
        echo "llama-server not up after ${SLM_WAIT_S}s -- starting anyway" \
             "(rule-based NLU will be used until it responds)."
        break
    fi
    sleep 2
done

stop=0
trap 'stop=1' INT TERM

while [ "$stop" -eq 0 ]; do
    echo "=== $(date -Is) starting chessmachine ===" | tee -a "$LOG_FILE"
    "$PYTHON" -m chessmachine --config config/config.yaml 2>&1 | tee -a "$LOG_FILE"
    status=${PIPESTATUS[0]}
    if [ "$stop" -eq 1 ]; then
        break
    fi
    echo "=== $(date -Is) chessmachine exited (status $status) --" \
         "restarting in ${RESTART_DELAY_S}s, Ctrl-C to stop ===" | tee -a "$LOG_FILE"
    sleep "$RESTART_DELAY_S"
done

echo "chessmachine autostart stopped."
