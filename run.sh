# run.sh - Production run script
#!/bin/bash

# run.sh - Start the Stable Diffusion API server
echo "Starting Stable Diffusion API..."

# Check if running in screen session
if [[ $STY ]]; then
    echo "Running in screen session: $STY"
else
    echo "Consider running in screen: screen -S sdapi ./run.sh"
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the server with logging
python app.py 2>&1 | tee logs/server.log
