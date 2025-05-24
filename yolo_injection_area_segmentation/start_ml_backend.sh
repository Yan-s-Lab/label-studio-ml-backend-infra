#!/bin/bash

# YOLO Injection Area Segmentation ML Backend Startup Script

echo "🚀 Starting YOLO Injection Area Segmentation ML Backend"
echo "=================================================="

# Clear potentially problematic environment variables
unset MODEL_DIR

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "📄 Loading configuration from .env file..."
    set -a  # automatically export all variables
    source .env
    set +a
else
    echo "⚠️  No .env file found, using default configuration..."
    echo "💡 Copy .env.example to .env and customize your settings"
fi

# Ensure MODEL_DIR is not set to problematic paths
if [ "$MODEL_DIR" = "/data/models" ] || [ "$MODEL_DIR" = "/data" ]; then
    echo "⚠️  Resetting MODEL_DIR from $MODEL_DIR to current directory for permission reasons"
    unset MODEL_DIR
fi

# Set default environment variables (fallback values)
export LABEL_STUDIO_URL="${LABEL_STUDIO_URL:-http://192.168.1.124:8080/}"
export LABEL_STUDIO_API_KEY="${LABEL_STUDIO_API_KEY:-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA1NTA4Mzk0NSwiaWF0IjoxNzQ3ODgzOTQ1LCJqdGkiOiI5MjM5MDgyYjVhNWY0MGFiYmYzNDQyNDFhMWQwOTUyNiIsInVzZXJfaWQiOjJ9.Me0AOzVfLcxBp83R5Ghjw90O-qbrXIdeoH1QE3F7l7E}"

# YOLO model settings
export CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.25}"
export IOU_THRESHOLD="${IOU_THRESHOLD:-0.7}"
export IMAGE_SIZE="${IMAGE_SIZE:-640}"
export DEVICE="${DEVICE:-auto}"
export MAX_DETECTIONS="${MAX_DETECTIONS:-300}"

# ML Backend settings
export LOG_LEVEL="${LOG_LEVEL:-DEBUG}"
export PORT="${PORT:-9090}"
export HOST="${HOST:-0.0.0.0}"
export WORKERS="${WORKERS:-1}"
export THREADS="${THREADS:-8}"

# Optional: Basic auth (uncomment if needed)
# export BASIC_AUTH_USER="your_username"
# export BASIC_AUTH_PASS="your_password"

# Validate required environment variables
if [ -z "$LABEL_STUDIO_URL" ]; then
    echo "❌ Error: LABEL_STUDIO_URL is required"
    echo "💡 Set it in .env file or as environment variable"
    exit 1
fi

if [ -z "$LABEL_STUDIO_API_KEY" ]; then
    echo "❌ Error: LABEL_STUDIO_API_KEY is required"
    echo "💡 Get it from Label Studio Account & Settings and set in .env file"
    exit 1
fi

echo "📋 Configuration:"
echo "  Label Studio URL: $LABEL_STUDIO_URL"
echo "  API Key: ${LABEL_STUDIO_API_KEY:0:20}..."
echo "  Confidence Threshold: $CONFIDENCE_THRESHOLD"
echo "  IoU Threshold: $IOU_THRESHOLD"
echo "  Image Size: $IMAGE_SIZE"
echo "  Device: $DEVICE"
echo "  Log Level: $LOG_LEVEL"
echo "  Port: $PORT"
echo "  Host: $HOST"
echo ""

# Check if model exists
MODEL_PATH="${MODEL_PATH:-train6/weights/best.pt}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found at $MODEL_PATH"
    echo "💡 Please ensure your trained YOLO model is in the correct location."
    echo "💡 Update MODEL_PATH in .env file if using a different path."
    exit 1
fi

echo "✅ Model file found: $MODEL_PATH"

# Check if port is available
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "❌ Error: Port $PORT is already in use"
    echo "💡 Use 'lsof -i :$PORT' to find the process using this port"
    echo "💡 Or change PORT in .env file"
    exit 1
fi

# Check if dependencies are installed
echo "🔍 Checking dependencies..."
python -c "import ultralytics, torch, cv2, numpy, PIL" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✅ Dependencies OK"

# Start the ML backend
echo ""
echo "🎯 Starting ML Backend..."
echo "   URL: http://$HOST:$PORT"
echo "   Health check: curl http://localhost:$PORT/health"
echo "   Label Studio integration: $LABEL_STUDIO_URL"
echo ""
echo "📊 Monitoring:"
echo "   - Watch logs for prediction requests"
echo "   - Check GPU usage with: nvidia-smi"
echo "   - Monitor system resources with: htop"
echo ""
echo "🛑 To stop the server: Press Ctrl+C"
echo ""

# Start the server
echo "🔧 Environment variables being passed to Python:"
echo "   LABEL_STUDIO_URL=$LABEL_STUDIO_URL"
echo "   LABEL_STUDIO_API_KEY=${LABEL_STUDIO_API_KEY:0:20}..."
echo "   LOG_LEVEL=$LOG_LEVEL"
echo ""

exec env LABEL_STUDIO_URL="$LABEL_STUDIO_URL" \
         LABEL_STUDIO_API_KEY="$LABEL_STUDIO_API_KEY" \
         CONFIDENCE_THRESHOLD="$CONFIDENCE_THRESHOLD" \
         IOU_THRESHOLD="$IOU_THRESHOLD" \
         IMAGE_SIZE="$IMAGE_SIZE" \
         DEVICE="$DEVICE" \
         MAX_DETECTIONS="$MAX_DETECTIONS" \
         LOG_LEVEL="$LOG_LEVEL" \
         python _wsgi.py --port $PORT --host $HOST --log-level $LOG_LEVEL

# python _wsgi.py --port 9090 --host 0.0.0.0 --log-level DEBUG