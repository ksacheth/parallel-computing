#!/usr/bin/env bash
# One-time setup of the ASGC parameter server on an Ubuntu (aarch64 or x86_64) VM.
# Run on the server from the repository root:  bash deploy/oracle/setup.sh
set -euo pipefail

PORT="${ASGC_PORT:-29500}"
BASE=/home/ubuntu/asgc
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

mkdir -p "$BASE/results" "$BASE/data"
if [ "$REPO_DIR" != "$BASE/repo" ]; then
  ln -sfn "$REPO_DIR" "$BASE/repo"
fi

if ! python3 -m venv --help >/dev/null 2>&1 || ! dpkg -s python3-venv >/dev/null 2>&1; then
  sudo apt-get update -y && sudo apt-get install -y python3-venv
fi
[ -d "$BASE/venv" ] || python3 -m venv "$BASE/venv"
"$BASE/venv/bin/pip" install -q --upgrade pip
# CPU-only wheels (aarch64 wheels exist on the PyTorch CPU index)
"$BASE/venv/bin/pip" install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
"$BASE/venv/bin/pip" install -q numpy pyyaml
"$BASE/venv/bin/pip" install -q -e "$REPO_DIR"

if [ ! -f "$BASE/ps.env" ]; then
  sed "s/^ASGC_AUTHKEY=.*/ASGC_AUTHKEY=$(openssl rand -hex 32)/" "$REPO_DIR/deploy/oracle/ps.env.example" > "$BASE/ps.env"
  chmod 600 "$BASE/ps.env"
  echo "generated $BASE/ps.env with a fresh authkey"
fi

# Host firewall: Oracle Ubuntu images end INPUT with a REJECT rule, so insert before it
if ! sudo iptables -C INPUT -p tcp --dport "$PORT" -m state --state NEW -j ACCEPT 2>/dev/null; then
  REJECT_LINE=$(sudo iptables -L INPUT --line-numbers -n | awk '$2=="REJECT"{print $1; exit}')
  sudo iptables -I INPUT "${REJECT_LINE:-1}" -p tcp --dport "$PORT" -m state --state NEW -j ACCEPT
fi
if command -v netfilter-persistent >/dev/null; then sudo netfilter-persistent save >/dev/null; fi

# Download datasets once so the first run does not stall on it
"$BASE/venv/bin/python" - <<PY
from torchvision import datasets
for cls in (datasets.FashionMNIST, datasets.CIFAR10):
    for train in (True, False):
        cls("$BASE/data", train=train, download=True)
PY

sudo cp "$REPO_DIR/deploy/oracle/asgc-ps.service" /etc/systemd/system/asgc-ps.service
sudo systemctl daemon-reload
echo
echo "Setup done. Also allow TCP $PORT ingress in the OCI VCN Security List."
echo "Edit $BASE/ps.env, then: sudo systemctl restart asgc-ps && journalctl -u asgc-ps -f"
