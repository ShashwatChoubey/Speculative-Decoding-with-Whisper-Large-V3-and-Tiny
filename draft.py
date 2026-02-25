"""In this code we use Whisper Tiny model to produce tokens till gamma (a hyperparameter we take to be 5) and then for each token we store its probability 
distribution that will be used in speculative decoding to compare to Large V3 model token probability distribution. Softmax is used to calculate the probability
distribution. If EOT token is generated it means the entire audio is done so we stop draft generation early."""


import torch

def decoder_step(model, tokens, encoder_output, device="cuda"):
    
    """Run one forward pass of the decoder
    Returns logits for next token prediction"""

    with torch.no_grad():
        tokens_tensor = torch.tensor([tokens], device=device)
        logits = model.decoder(tokens_tensor, encoder_output)
        last_logits = logits[0, -1, :]
        
        return last_logits


def logits_to_probs(logits):
    
    """Convert raw logits to probability distribution using softmax"""
    
    
    return torch.nn.functional.softmax(logits, dim=-1)


def generate_draft(
    draft_model,
    tokenizer,
    tiny_encoded,
    all_tokens,
    gamma=5,
    device="cuda"
):
    
    """Generate gamma draft tokens using Whisper Tiny"""
    
    
    
    current_tokens = list(tokenizer.sot_sequence) + all_tokens
    
    draft_tokens = []
    draft_probs = []
    
    for _ in range(gamma):
       
        logits = decoder_step(
            draft_model,
            current_tokens,
            tiny_encoded,
            device
        )
        
        
        
       
        probs = logits_to_probs(logits)
        
        
        next_token = torch.argmax(probs).item()
        
        draft_tokens.append(next_token)
        draft_probs.append(probs)
        current_tokens.append(next_token)

        if next_token == tokenizer.eot:
            break
    
    return draft_tokens, draft_probs
