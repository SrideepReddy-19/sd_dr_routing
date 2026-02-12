#!/bin/bash

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$PROJECT_DIR/data/logs"

mkdir -p "$LOG_DIR"

echo "Starting Ryu Controller..."

nohup $PROJECT_DIR/venv/bin/ryu-manager \
$PROJECT_DIR/controller/ryu_controller.py \
--ofp-tcp-listen-port 6633 \
--observe-links > "$LOG_DIR/ryu.log" 2>&1 &

sleep 5

echo "Starting Mininet Topology..."

nohup $PROJECT_DIR/venv/bin/python3 -c "
from topology.custom_topology import create_network
net = create_network('127.0.0.1', 6633)
net.start()
import time
while True:
    time.sleep(60)
" > "$LOG_DIR/mininet.log" 2>&1 &

sleep 5

echo "Starting Dashboard on http://localhost:9000"

export PYTHONPATH=$PROJECT_DIR
$PROJECT_DIR/venv/bin/python3 -m visualization.dashboard --port 9000
