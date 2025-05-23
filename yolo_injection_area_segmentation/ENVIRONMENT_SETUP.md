# Environment Setup Guide

This guide explains how to set up environment variables for the YOLO Injection Area Segmentation ML Backend.

## Required Environment Variables

### Essential Variables

```bash
# Label Studio connection
LABEL_STUDIO_URL=http://192.168.1.124:8080/
LABEL_STUDIO_API_KEY=your_api_key_here

# YOLO model settings
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.7
IMAGE_SIZE=640
DEVICE=auto
```

### Optional Variables

```bash
# ML Backend server settings
LOG_LEVEL=INFO
WORKERS=1
THREADS=8
MAX_DETECTIONS=300

# Basic authentication (if needed)
BASIC_AUTH_USER=username
BASIC_AUTH_PASS=password
```

## Setup Methods

### Method 1: Using Docker Compose (Recommended)

1. **Edit the `.env` file:**
```bash
nano .env
```

2. **Update your values:**
```bash
LABEL_STUDIO_URL=http://192.168.1.124:8080/
LABEL_STUDIO_API_KEY=your_actual_api_key
CONFIDENCE_THRESHOLD=0.25
```

3. **Start with Docker:**
```bash
docker-compose up
```

### Method 2: Using the Startup Script

1. **Edit the startup script:**
```bash
nano start_ml_backend.sh
```

2. **Update the environment variables in the script**

3. **Run the script:**
```bash
./start_ml_backend.sh
```

### Method 3: Manual Environment Setup

**For Linux/Mac:**
```bash
export LABEL_STUDIO_URL="http://192.168.1.124:8080/"
export LABEL_STUDIO_API_KEY="your_api_key"
export CONFIDENCE_THRESHOLD="0.25"
export IOU_THRESHOLD="0.7"
export IMAGE_SIZE="640"
export DEVICE="auto"

python _wsgi.py --port 9090
```

**For Windows (PowerShell):**
```powershell
$env:LABEL_STUDIO_URL="http://192.168.1.124:8080/"
$env:LABEL_STUDIO_API_KEY="your_api_key"
$env:CONFIDENCE_THRESHOLD="0.25"
$env:IOU_THRESHOLD="0.7"
$env:IMAGE_SIZE="640"
$env:DEVICE="auto"

python _wsgi.py --port 9090
```

**For Windows (Command Prompt):**
```cmd
set LABEL_STUDIO_URL=http://192.168.1.124:8080/
set LABEL_STUDIO_API_KEY=your_api_key
set CONFIDENCE_THRESHOLD=0.25
set IOU_THRESHOLD=0.7
set IMAGE_SIZE=640
set DEVICE=auto

python _wsgi.py --port 9090
```

### Method 4: Using Python-dotenv

1. **Install python-dotenv:**
```bash
pip install python-dotenv
```

2. **Create a `.env` file with your variables**

3. **Load in your Python code:**
```python
from dotenv import load_dotenv
load_dotenv()
```

## Getting Your Label Studio API Key

1. **Open Label Studio** at `http://192.168.1.124:8080/`
2. **Go to Account & Settings** (click your profile icon)
3. **Click on "Access Token"**
4. **Copy the token** and use it as `LABEL_STUDIO_API_KEY`

## Verifying Environment Setup

### Check Environment Variables
```bash
python -c "import os; print('LABEL_STUDIO_URL:', os.getenv('LABEL_STUDIO_URL')); print('API_KEY set:', bool(os.getenv('LABEL_STUDIO_API_KEY')))"
```

### Test ML Backend Connection
```bash
# Start the ML backend
python _wsgi.py --port 9090

# In another terminal, test the connection
curl http://localhost:9090/health
```

### Test Label Studio Connection
```bash
curl -H "Authorization: Token your_api_key" http://192.168.1.124:8080/api/projects/
```

## Troubleshooting

### Common Issues

1. **"Connection refused" error:**
   - Check if Label Studio is running at the specified URL
   - Verify the IP address and port

2. **"Unauthorized" error:**
   - Check if the API key is correct
   - Ensure the API key hasn't expired

3. **"Model not found" error:**
   - Verify `train6/weights/best.pt` exists
   - Check file permissions

4. **Import errors:**
   - Install dependencies: `pip install -r requirements.txt`

### Environment Variable Priority

The system reads environment variables in this order:
1. System environment variables
2. `.env` file
3. Default values in `config.py`

## Security Notes

- **Never commit API keys** to version control
- **Use `.env` files** for local development
- **Use environment variables** in production
- **Consider using secrets management** for production deployments

## Example Complete Setup

```bash
# 1. Clone and navigate to directory
cd yolo_injection_area_segmentation

# 2. Create .env file
cat > .env << EOF
LABEL_STUDIO_URL=http://192.168.1.124:8080/
LABEL_STUDIO_API_KEY=your_actual_api_key_here
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.7
IMAGE_SIZE=640
DEVICE=auto
LOG_LEVEL=INFO
EOF

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the service
docker-compose up
# OR
./start_ml_backend.sh
# OR
python _wsgi.py --port 9090
```
