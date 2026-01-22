
import torch
from transformers import AutoTokenizer

def test_log_prob_logic():
    # Use GPT2 as a proxy for BPE behavior (it merges close tokens)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Simulate the prompt and response construction
    # Scenario: Prompt ends with double newline (common in chat templates), Response starts with a word.
    
    # Case 1: Safe boundary (usually)
    prompt_safe = "User: Hello\nAssistant:\n\n"
    response_safe = "Sure, I can help."
    
    # Case 2: Dangerous boundary (Merging)
    # GPT2 merges "The" and "cat" -> "Thecat"? No.
    # But "The" and " cat" (space) -> "The" and " cat".
    # What if prompt ends with "The" and response starts with "cat"?
    prompt_merge = "The" 
    response_merge = "cat"
    
    # Case 3: Newline merging
    # Prompt: "Hello\n"
    # Response: "\nWorld"
    prompt_newline = "Hello\n"
    response_newline = "\nWorld"

    def check_case(name, p, r):
        print(f"--- Check Case: {name} ---")
        print(f"Prompt: {repr(p)}")
        print(f"Response: {repr(r)}")
        
        # Simulate _get_log_probs logic
        # 1. Tokenize Prompt
        p_inputs = tokenizer(p, add_special_tokens=False)
        p_ids = p_inputs['input_ids']
        p_len = len(p_ids)
        print(f"Prompt Tokens: {tokenizer.convert_ids_to_tokens(p_ids)} (Len: {p_len})")
        
        # 2. Tokenize Full (P + R)
        full_text = p + r
        full_inputs = tokenizer(full_text, add_special_tokens=False)
        full_ids = full_inputs['input_ids']
        full_len = len(full_ids)
        print(f"Full Tokens:   {tokenizer.convert_ids_to_tokens(full_ids)} (Len: {full_len})")
        
        # 3. Simulate Mask Construction
        # We assume full_ids are [PromptTokens, ResponseTokens]
        # So Prompt ends at index p_len - 1. Response starts at p_len.
        
        # Check ALIGNMENT
        # Does full_ids[:p_len] match p_ids?
        match_prompt = (full_ids[:p_len] == p_ids)
        print(f"Prompt Match in Full: {match_prompt}")
        
        if not match_prompt:
            print("❌ MISMATCH DETECTED!")
            print(f"Expected Prompt IDs: {p_ids}")
            print(f"Actual Start IDs:    {full_ids[:p_len]}")
        else:
            print("✅ Alignment OK (Start matches)")
            
        # Check LOGIC of mask
        # Code uses p_len as the start of response.
        # If full_len < p_len + r_len (merging happened), does p_len start response correctly?
        
        r_inputs = tokenizer(r, add_special_tokens=False)
        r_ids = r_inputs['input_ids']
        r_len = len(r_ids)
        print(f"Response Tokens (isolated): {tokenizer.convert_ids_to_tokens(r_ids)}")
        
        if full_len != p_len + r_len:
            print(f"⚠️ Length Mismatch! Full: {full_len}, P+R: {p_len + r_len}")
            # Identify where the merge happened
            # If mismatch, usually full_ids[p_len-1] is DIFFERENT from p_ids[-1]
            if full_ids[p_len-1] != p_ids[-1]:
                 print(f"   -> Merge happened at boundary index {p_len-1} (Last prompt token changed)")
            elif full_ids[p_len] != r_ids[0]:
                 print(f"   -> Merge happened at boundary index {p_len} (First response token changed or shifted)")
        else:
            print("✅ Length Match (No boundary merging)")

    check_case("Safe Boundary", prompt_safe, response_safe)
    print("\n")
    check_case("Forced Merge", prompt_merge, response_merge)
    print("\n")
    check_case("Newline Merge", prompt_newline, response_newline)

if __name__ == "__main__":
    test_log_prob_logic()
