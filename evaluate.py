""""In this code we use Whisper original transcribe as our baseline reference and run our SpeculativeWhisper on the same audio. WER (Word Error Rate) is 
calculated as sum of substitutions, deletions and insertions divided by total number of words. We calculate WER of speculative transcript against the 
standard transcript as reference and also compare time taken by both. Lower WER means speculative decoding is closer to standard output."""


import time
import whisper
import torch
from jiwer import wer

from audio_utils import load_audio_as_mel, encode_audio
from speculative_whisper import SpeculativeWhisper


def run_standard_transcription(model, audio_path):
    
    """Run standard Whisper Large V3 greedy decoding
    Used as baseline for comparison"""
    
    start = time.time()
    result = model.transcribe(audio_path)
    elapsed = time.time() - start

    return result["text"].strip(), elapsed


def run_speculative_transcription(sw, audio_path, gamma=5, max_tokens=500):
    
    """Run speculative decoding transcription"""

    
    
    start = time.time()
    results = sw.transcribe(
        audio_files=[audio_path],
        gamma=gamma,
        max_tokens=max_tokens
    )
    elapsed = time.time() - start

    return results[0], elapsed


def calculate_wer(reference, hypothesis):
    
    """Calculate Word Error Rate between reference and hypothesis"""

    
    
    error_rate = wer(
        reference.lower().strip(),
        hypothesis.lower().strip()
    )
    return error_rate


def evaluate(audio_path, gamma=5, max_tokens=500, device="cuda"):
    
    """Full evaluation — compare speculative decoding vs standard"""

    
    print("EVALUATION: Speculative Decoding vs Standard Decoding")

    
    sw = SpeculativeWhisper(
        draft_model="tiny",
        final_model="large-v3",
        device=device
    )

    
    print("\n1. Running Standard Large V3...")
    standard_text, standard_time = run_standard_transcription(
        sw.final_model,
        audio_path
    )
    print(f"Time: {standard_time:.2f}s")
    print(f"Transcript: {standard_text[:100]}...")

    # ---- Speculative Decoding ----
    print("\n2. Running Speculative Decoding...")
    speculative_text, speculative_time = run_speculative_transcription(
        sw,
        audio_path,
        gamma=gamma,
        max_tokens=max_tokens
    )
    print(f"Time: {speculative_time:.2f}s")
    print(f"Transcript: {speculative_text[:100]}...")

    # ---- WER Calculation ----
    error_rate = calculate_wer(standard_text, speculative_text)
    speedup = standard_time / speculative_time

    # ---- Results ----
   
    print("RESULTS SUMMARY")
    print(f"Standard Large V3 time:     {standard_time:.2f}s")
    print(f"Speculative decoding time:  {speculative_time:.2f}s")
    print(f"Speedup:                    {speedup:.2f}x")
    print(f"WER:                        {error_rate*100:.2f}%")
    print(f"\nStandard transcript:\n{standard_text}")
    print(f"\nSpeculative transcript:\n{speculative_text}")
   

    return {
        "standard_time": standard_time,
        "speculative_time": speculative_time,
        "speedup": speedup,
        "wer": error_rate,
        "standard_transcript": standard_text,
        "speculative_transcript": speculative_text
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    results = evaluate(audio_path)
