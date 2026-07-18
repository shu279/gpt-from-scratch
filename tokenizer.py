from collections import Counter

# Replace the most frequently occurring pairs of consecutive bytes with new id
def merge(ids, pair, new_id):
    res = []
    i = 0
    while i < len(ids):
        if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            res.append(new_id)
            i += 2
        else:
            res.append(ids[i])
            i += 1
    return res

# Simple byte-pair encoding BPE
# Output dict of key = ids to merge, value = new id
def train_bpe(text, num_merges):
    ids = list(text.encode("utf-8"))
    new_id = 256
    merges = {}

    for _ in range(num_merges):
        pairs = Counter(zip(ids, ids[1:]))

        if not pairs: break

        best_pair = max(pairs, key=pairs.get)
        ids = merge(ids, best_pair, new_id)
        merges[best_pair] = new_id
        new_id += 1
    return merges

# Encode train / validation to token ID vector
def encode(text, merges):
    ids = list(text.encode("utf-8"))

    for pair, token_id in merges.items():
        ids = merge(ids, pair, token_id)
    return ids

def decode(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for pair, token_id in merges.items():
        vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]

    res = b"".join(vocab[token_id] for token_id in ids)

    return res.decode("utf-8")
