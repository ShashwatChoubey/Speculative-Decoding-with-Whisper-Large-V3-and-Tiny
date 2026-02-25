"""In this code we first generate probability distribution of each draft token using Large V3 model in ONE forward pass, not the token itself but its probability distribution. Large V3 distribution is p(x) and Tiny distribution is q(x). We calculate acceptance ratio = p/q and sample u from uniform distribution, if min(1, ratio) >= u the token is accepted and optionally we can also generate the next token from Large V3. If rejected we sample a corrected token from adjusted distribution clamp(p-q, min=0) and stop accepting further draft tokens. Then Large V3 predicts the next token and the whole draft generation process starts again."""

import torch
from draft import logits_to_probs


def get_large_probs(
    final_model,
    tokenizer,
    all_tokens,
    draft_tokens,
    large_encoded,
    device="cuda"
):
    
    """Run Large V3 in ONE forward pass on draft sequence
    Returns p(x) for each draft token position"""
    
    with torch.no_grad():
        # Full sequence = SOT + accepted tokens + draft tokens
        full_sequence = (
            list(tokenizer.sot_sequence) +
            all_tokens +
            draft_tokens
        )
        tokens_tensor = torch.tensor(
            [full_sequence], device=device
        )
        
        
        logits = final_model.decoder(
            tokens_tensor, large_encoded
        )
        
       
        large_probs = []
        draft_start = (
            len(tokenizer.sot_sequence) + len(all_tokens)
        )
        
        for i in range(len(draft_tokens)):
            pos_logits = logits[0, draft_start + i-1 , :]
            probs = logits_to_probs(pos_logits)
            large_probs.append(probs)
        
        return large_probs


def rejection_sampling(draft_tokens, draft_probs, large_probs):
    
    """Accept or reject draft tokens based on p(x) vs q(x)"""
    
   
    
    accepted_tokens = []
    
    for token, q_probs, p_probs in zip(
        draft_tokens, draft_probs, large_probs
    ):
        
        q = q_probs[token].item()
        p = p_probs[token].item()
        
        # Acceptance ratio
        ratio = p / (q + 1e-10)
        
       
        u = torch.rand(1).item()
        
        if min(1, ratio) >= u:
            # Token accepted
            accepted_tokens.append(token)
        else:
            # Token rejected 
            min_vocab = min(p_probs.shape[0], q_probs.shape[0])
            adjusted = torch.clamp(
                p_probs[:min_vocab] - q_probs[:min_vocab], min=0
            )
            adjusted = adjusted / (adjusted.sum() + 1e-10)
            corrected = torch.multinomial(adjusted, 1).item()
            accepted_tokens.append(corrected)
            break
    
    return accepted_tokens


def get_next_token(
    final_model,
    tokenizer,
    current_tokens,
    large_encoded,
    device="cuda"
):
    
    """Get next token from Large V3 after accepted sequence"""
    
    
    
    with torch.no_grad():
        full_sequence = (
            list(tokenizer.sot_sequence) + current_tokens
        )
        tokens_tensor = torch.tensor(
            [full_sequence], device=device
        )
        
        logits = final_model.decoder(
            tokens_tensor, large_encoded
        )
        
        last_logits = logits[0, -1, :]
        probs = logits_to_probs(last_logits)
        
        return torch.argmax(probs).item()