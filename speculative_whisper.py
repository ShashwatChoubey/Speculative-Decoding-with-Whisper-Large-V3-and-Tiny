"""In this code we define the SpeculativeWhisper class which brings together all the previous functions to give the final output. We load both Tiny and Large V3 models and the tokenizer. For each audio file we convert it to mel spectrogram, encode it with both models, then run the speculative decoding loop which keeps calling draft and verify until EOT is generated or max tokens is reached. After decoding we remove special tokens like SOT and EOT from the final token list as they cause problems in the output text. We also support multiple audio files by looping through them one by one."""

import torch
import time
import whisper

from audio_utils import load_audio_as_mel, encode_audio
from draft import generate_draft
from verify import get_large_probs, rejection_sampling, get_next_token


class SpeculativeWhisper:
    def __init__(
        self,
        draft_model="tiny",
        final_model="large-v3",
        device="cuda"
    ):
        
        """Initialize SpeculativeWhisper with draft and final models"""

        
        
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
       
        

        print(f"Loading draft model: {draft_model}...")
        self.draft_model = whisper.load_model(
            draft_model
        ).to(self.device)

        print(f"Loading final model: {final_model}...")
        self.final_model = whisper.load_model(
            final_model
        ).to(self.device)

        
        self.tokenizer = whisper.decoding.get_tokenizer(
            multilingual=self.final_model.is_multilingual,
            language="en",
            task="transcribe"
        )

        print("Models loaded successfully")

    def _clean_transcript(self, tokens):
        
        """Remove special tokens from final output"""

        special_ids = set(
            list(self.tokenizer.sot_sequence) +
            [self.tokenizer.eot, self.tokenizer.sot]
        )
        clean = [t for t in tokens if t not in special_ids]
        return self.tokenizer.decode(clean).strip()

    def _speculative_decode_single(
        self,
        tiny_encoded,
        large_encoded,
        gamma=5,
        max_tokens=500
    ):
        
        """Full speculative decoding loop for a single audio"""

        
        
        all_tokens = []

        while len(all_tokens) < max_tokens:

            # ---- DRAFT PHASE ----
           
            draft_tokens, draft_probs = generate_draft(
                draft_model=self.draft_model,
                tokenizer=self.tokenizer,
                tiny_encoded=tiny_encoded,
                all_tokens=all_tokens,
                gamma=gamma,
                device=self.device
            )

            # ---- VERIFY PHASE ----
            
            large_probs = get_large_probs(
                final_model=self.final_model,
                tokenizer=self.tokenizer,
                all_tokens=all_tokens,
                draft_tokens=draft_tokens,
                large_encoded=large_encoded,
                device=self.device
            )

            # ---- REJECTION SAMPLING ----
           
            accepted = rejection_sampling(
                draft_tokens=draft_tokens,
                draft_probs=draft_probs,
                large_probs=large_probs
            )
            all_tokens.extend(accepted)

            
            if self.tokenizer.eot in accepted:
                break

            
            next_token = get_next_token(
                final_model=self.final_model,
                tokenizer=self.tokenizer,
                current_tokens=all_tokens,
                large_encoded=large_encoded,
                device=self.device
            )

            
            if next_token == self.tokenizer.eot:
                break

            all_tokens.append(next_token)

        return self._clean_transcript(all_tokens)

    def transcribe(
        self,
        audio_files,
        gamma=5,
        max_tokens=500,
        batch_size=1
    ):
        
        """Transcribe multiple audio files using speculative decoding"""

        
        
        # Handle single file input
        if isinstance(audio_files, str):
            audio_files = [audio_files]

        results = []
        total_start = time.time()

        print(f"Transcribing {len(audio_files)} file(s)...")
        print("=" * 50)

        # Sequential batching
        for i, audio_path in enumerate(audio_files):
            print(f"\n[{i+1}/{len(audio_files)}] Processing: {audio_path}")
            start = time.time()

            
            mel_tiny, mel_large = load_audio_as_mel(
                audio_path, self.device
            )
            tiny_encoded, large_encoded = encode_audio(
                self.draft_model,
                self.final_model,
                mel_tiny,
                mel_large,
                
                
            )

            
            transcript = self._speculative_decode_single(
                tiny_encoded=tiny_encoded,
                large_encoded=large_encoded,
                gamma=gamma,
                max_tokens=max_tokens
            )

            elapsed = time.time() - start
            print(f"Time: {elapsed:.2f}s")
            print(f"Transcript: {transcript[:100]}...")

            results.append(transcript)

        total_time = time.time() - total_start
        print(f"\n{'='*50}")
        print(
            f"Total time for {len(audio_files)} "
            f"file(s): {total_time:.2f}s"
        )

        return results