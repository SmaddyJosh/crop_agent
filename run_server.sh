#!/bin/bash
echo "Starting the Crop Vision Web API..."
python3 -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
