#!/usr/bin/env python3
import argparse
from classifier_pycoral import Classifier
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Test the bird audio classifier')
    parser.add_argument('--audio', required=True, help='Path to the audio file to classify')
    parser.add_argument('--model', required=True, help='Path to the MLK model')
    parser.add_argument('--sample-rate', type=int, default=48000, help='Sample rate (default: 48000)')
    parser.add_argument('--clip-duration', type=float, default=3.0, help='Clip duration in seconds (default: 3.0)')
    parser.add_argument('--overlap', type=float, default=1.0, help='Overlap between clips (default: 1.0)')
    
    args = parser.parse_args()
    
    # Initialize classifier
    classifier = Classifier(
        path_to_mlk_model=args.model,
        sr=args.sample_rate,
        clip_dur=args.clip_duration
    )
    
    # Run classification
    print(f"\nProcessing audio file: {args.audio}")
    predictions, timestamps = classifier.classify(args.audio, overlap=args.overlap)
    
    # Print results
    print("\nClassification Results:")
    print("-" * 50)
    for pred, t in zip(predictions, timestamps):
        print(f"Time: {t:.2f}s - Confidence: {pred:.4f}")
    
    if isinstance(predictions, list):
        print(f"\nMaximum confidence score: {max(predictions):.4f}")

if __name__ == "__main__":
    main()
