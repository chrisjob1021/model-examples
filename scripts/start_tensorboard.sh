#!/bin/bash

# TensorBoard startup script for model-examples project
# This script starts TensorBoard to visualize training logs

# Default settings
DEFAULT_LOGDIR="./logs"
DEFAULT_PORT=6006
DEFAULT_HOST="0.0.0.0"

# Parse command line arguments
LOGDIR=${1:-$DEFAULT_LOGDIR}
PORT=${2:-$DEFAULT_PORT}
HOST=${3:-$DEFAULT_HOST}

echo "🚀 Starting TensorBoard..."
echo "📂 Log directory: $LOGDIR"
echo "🌐 Host: $HOST"
echo "🔌 Port: $PORT"
echo "🔗 URL: http://$HOST:$PORT"
echo ""

# Check if log directory exists
if [ ! -d "$LOGDIR" ]; then
    echo "❌ Warning: Log directory '$LOGDIR' does not exist"
    echo "   TensorBoard will start anyway and monitor for new logs"
    echo ""
fi

# Check if tensorboard is installed
if ! command -v tensorboard &> /dev/null; then
    echo "❌ Error: TensorBoard is not installed"
    echo "   Install with: pip install tensorboard"
    exit 1
fi

echo "⏳ Starting TensorBoard server..."
echo "   Press Ctrl+C to stop"
echo ""

# Start TensorBoard
tensorboard --logdir="$LOGDIR" --host="$HOST" --port="$PORT"