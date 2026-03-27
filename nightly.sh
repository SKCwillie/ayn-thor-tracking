#!/usr/bin/env bash
set -e

cd /home/ubuntu/ayn-thor-tracking

echo "Starting nightly update: $(date)"

source .venv/bin/activate

python main.py

echo "Restarting service"
sudo systemctl restart thor-tracking

echo "Nightly update complete: $(date)"