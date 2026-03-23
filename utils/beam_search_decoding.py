import numpy as np

def beam_search_decode(log_probs: np.ndarray, beam_width: int, top_k: int = 1) -> list:
    """
    Perform beam search decoding over a sequence of log-probability distributions.
    
    Args:
        log_probs: numpy array of shape (T, V) with log-probabilities at each step
        beam_width: number of beams to maintain at each step
        top_k: number of top sequences to return
    
    Returns:
        List of (sequence, score) tuples sorted by score descending
    """
    steps = log_probs.shape[0]
    results = []
    curr_result = [([idx], score) for idx, score in enumerate(log_probs[0])]
    curr_result.sort(key = lambda x : x[1], reverse = True)
    curr_result = curr_result[:beam_width]
    for step in range(1, steps):
        next_beam = []
        while len(curr_result) > 0:
            seq, score = curr_result.pop(0)
            for idx, log_prob in enumerate(log_probs[step]):
                next_seq = seq + [idx]
                next_score = score + log_prob
                next_beam.append((next_seq,  next_score))
        next_beam.sort(key = lambda x : x[1], reverse = True)
        curr_result = next_beam[:beam_width]
    return curr_result[:top_k]

import numpy as np

def beam_search(model_func, initial_input, beam_width=5, max_len=50, alpha=0.7):
    """
    Args:
        model_func: Function that takes current sequence and returns log_probs for next step.
        initial_input: The starting token (or image embedding).
        alpha: Length normalization coefficient (usually 0.6 - 0.7).
    """
    # Beams stored as: (normalized_score, cumulative_log_prob, sequence)
    beams = [(0.0, 0.0, [initial_input])]
    completed_sequences = []

    for step in range(max_len):
        candidates = []
        
        for norm_score, total_log_prob, seq in beams:

            # This is the log probabilities of the next patch tokens
            next_token_log_probs = model_func(seq) 
            
            # Find top candidates for this beam
            top_indices = np.argsort(next_token_log_probs)[-beam_width:]
            
            for idx in top_indices:
                new_seq = seq + [idx]
                new_total_log_prob = total_log_prob + next_token_log_probs[idx]
                
                # Length Normalization Formula
                # Score = Total_Log_Prob / (Length ^ alpha)
                lp = ((5 + len(new_seq))**alpha) / ((5 + 1)**alpha)
                new_norm_score = new_total_log_prob / lp
                
                if idx == 0:  # This is the <EOS> token
                    completed_sequences.append((new_norm_score, new_seq))
                else:
                    candidates.append((new_norm_score, new_total_log_prob, new_seq))
        
        # Pruning: Keep only the best 'beam_width' candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
        
        # Early Stopping: If we found enough completed paths
        if len(completed_sequences) >= beam_width:
            break

    # Final sort including completed paths
    completed_sequences.sort(key=lambda x: x[0], reverse=True)
    return completed_sequences[0] if completed_sequences else beams[0]

if __name__ == "__main__":
    log_probs = np.array([[-0.5, -1.0, -2.0], [-0.8, -0.3, -1.5]])
    result = beam_search_decode(log_probs, beam_width=2, top_k=1)
    print(result)