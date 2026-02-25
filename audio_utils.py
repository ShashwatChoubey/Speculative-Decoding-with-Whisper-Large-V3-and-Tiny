"""In whisper architecture before the data is passed to the encoder it is converted to mel log spectrogram which is a freq vs time representation in which
freq is on mel scale which mimics human hearing. The values represent log energy (in dB) and by color we show the amplitude. In this code we take our audio data
and convert it to mel spectrogram to feed it into whisper. Tiny needs 80 mel bins and Large V3 needs 128 mel bins."""

import whisper
import torch


def load_audio_as_mel(audio_path: str, device: str = "cuda"):
    
    """ Load audio file and convert to log-mel spectrograms
    Returns separate mels for tiny (80 bins) and large (128 bins)"""
   
    audio = whisper.load_audio(audio_path)
    
    
    audio = whisper.pad_or_trim(audio)
    
    # Tiny needs 80 mel bins
    mel_tiny = whisper.log_mel_spectrogram(
        audio, n_mels=80
    ).to(device)
    
    # Large V3 needs 128 mel bins
    mel_large = whisper.log_mel_spectrogram(
        audio, n_mels=128
    ).to(device)
    
    return mel_tiny, mel_large


def encode_audio(draft_model, final_model, mel_tiny, mel_large):
    
    """Encode mel spectrograms with both models"""
    
    with torch.no_grad():
        tiny_encoded = draft_model.encoder(
            mel_tiny.unsqueeze(0)
        )
        large_encoded = final_model.encoder(
            mel_large.unsqueeze(0)
        )
    
    return tiny_encoded, large_encoded
