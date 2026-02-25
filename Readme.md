# Speculative Whisper

Speculative decoding is a technique to increase inference speed. It uses a small model (Whisper Tiny) to generate tokens till gamma and also stores the probability distribution of each token. Then it uses a single forward pass of the large model (Whisper Large V3) to produce probability distribution for those same tokens. We compare probability distribution of small to large, if large probability is greater the token is accepted, if not it is rejected and all tokens after it are discarded. We then sample a corrected token from the large model and start the process again with the small model. This continues till end of transcript.

## How It Works

Audio file goes in and is converted to mel spectrogram. Whisper Tiny generates draft tokens and their probability distributions. Whisper Large V3 verifies those tokens in a single forward pass and accepts or rejects them. Final transcript is built from accepted tokens.


## File Structure

speculative_whisper.py - Main class that brings everything together  
draft.py - Draft token generation using Whisper Tiny  
verify.py - Verification and rejection sampling using Whisper Large V3  
audio_utils.py - Audio loading and mel spectrogram conversion  
evaluate.py - Speed and WER comparison  
api.py - use fastapi to expose speculative_whisper.py

### Installation

Clone the repository
git clone <your_repo_link>
cd speculative_whisper
Install dependencies
pip install openai-whisper jiwer fastapi uvicorn

### Usage

Run evaluation to compare speculative decoding vs standard Large V3:
python evaluate.py your_audio.mp3

### Results

Tested on GPU (T4 Colab):

Speedup: 3-5x on longer audio

WER: 10.53% on clean audio
