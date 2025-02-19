#!/bin/bash

# Example paths - replace these with your actual paths
AUDIO_FILE="/home/jkor/my_services/deep-chirp-monitor/recordings/audio_chunk_20250216_192046.wav"
MODEL_PATH="../models/model_v3_5.keras"

# Run the classifier with default parameters
echo "Running classifier with default parameters..."
uv run test_classifier.py \
    --audio "$AUDIO_FILE" \
    --model "$MODEL_PATH"

# Run the classifier with custom parameters
# echo -e "\nRunning classifier with custom parameters..."
# uv run POC/test_classifier.py \
#     --audio "$AUDIO_FILE" \
#     --model "$MODEL_PATH" \
#     --sample-rate 44100 \
#     --clip-duration 2.5 \
#     --overlap 0.5
c