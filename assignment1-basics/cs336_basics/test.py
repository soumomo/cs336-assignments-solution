import os
from tokenizer import train_bpe_tokenizer

if __name__ == "__main__":
    # 1. Create a dummy corpus file for testing
    sample_text = "hug bug hugger hugger bugger"
    corpus_filename = "test_corpus.txt"

    with open(corpus_filename, "w", encoding="utf-8") as f:
        f.write(sample_text)

    # 2. Define test configuration hyper-parameters
    vocab_size = 265  # 256 basic bytes + 2 special tokens + 7 merges
    special_tokens = ["<pad>", "<unk>"]

    print("--- Starting BPE Tokenizer Training ---")

    try:
        # 3. Call your BPE function
        vocab, merges = train_bpe_tokenizer(
            input_path=corpus_filename, 
            vocab_size=vocab_size, 
            special_tokens=special_tokens
        )

        # 4. Display the results
        print(f"\nTraining Successful!")
        print(f"Total Merges performed: {len(merges)}")
        print(f"Final Vocabulary Size: {len(vocab)}")
        
        print("\n--- Displaying New Merged Tokens ---")
        # Tokens starting from ID 258 are the newly learned subwords
        for token_id in range(256 + len(special_tokens), len(vocab)):
            # Decode the bytes back to a readable string format
            readable_token = vocab[token_id].decode('utf-8', errors='replace')
            print(f"ID {token_id}: '{readable_token}'")

    finally:
        # 5. Cleanup the temporary text file
        if os.path.exists(corpus_filename):
            os.remove(corpus_filename)