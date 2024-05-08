import torch
import torch.nn.functional as F


def prev_seq_similarity(typicality_sequence, glove_model):
    """
    Get the similarity of each exemplar to the previous exemplar using non-contextual GloVe embedding
    """
    sim = [float('inf')]
    for i, resp in enumerate(typicality_sequence[1:]):
        try:
            sim += [F.cosine_similarity(torch.Tensor(glove_model[typicality_sequence[i].lower()]), torch.Tensor(glove_model[typicality_sequence[i-1].lower()]), dim=0)]
        except KeyError as e:
            # print(f"WARNING: Key {e} does not exist in GloVe! Using inf...")
            sim += [float('inf')]
    return sim


def glove_pairwise_distance(typicality_sequence, glove_model):
    """
    Get the pairwise GloVe distance between a list of exemplars
    """
    typicality_sequence = [t.lower() for t in typicality_sequence if t.lower() in glove_model.keys()]

    embedding = []
    for i, item in enumerate(typicality_sequence):
        embedding += [glove_model[item.lower()]]

    e1l = []
    for e1 in embedding:
        e2l = []
        for e2 in embedding:
            cosine_distance = 1 - F.cosine_similarity(torch.Tensor(e1), torch.Tensor(e2), dim=0)
            e2l += [float(cosine_distance)]
        e1l += [e2l]

    distances = e1l

    return distances


def glove_pairwise_similarity(typicality_sequence, glove_model):
    dist = glove_pairwise_distance(typicality_sequence, glove_model)
    for i in range(len(dist)):
        for j in range(len(dist)):
            dist[i][j] = 1 - dist[i][j]
    return dist


def bert_pairwise_distance(typicality_sequence, model, tokenizer):
    """
    Get the pairwise BERT distance between a list of exemplars
    """
    typicality_sequence = [t.lower() for t in typicality_sequence]
    typicality_str = ' '.join(typicality_sequence)

    tokenized = tokenizer.encode(typicality_str, add_special_tokens=True)

    print(f'{len(typicality_sequence)} categories were tokenized into {len(tokenized)} subwords')

    # Get indices of each item's tokens
    item_indices = []
    total_tokens = 0
    for i, item in enumerate(typicality_sequence):
        start_idx = item_indices[i - 1][1] if i != 0 else 0
        num_tokens = len(tokenizer.tokenize(item))
        total_tokens += num_tokens
        item_indices += [(start_idx, start_idx + num_tokens)]
    assert len(tokenized) == total_tokens + 2, "Tokenized sequence length should align to items tokenized invididually"

    # Padding (for multiple batches)
    # max_len = 0
    # for i in tokenized.values:
    #     if len(i) > max_len:
    #         max_len = len(i)
    # padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    # attention_mask = np.where(padded != 0, 1, 0)
    # input_ids = torch.tensor(padded)
    # attention_mask = torch.tensor(attention_mask)

    input_ids = torch.tensor([tokenized])

    with torch.no_grad():
        encoding = model(input_ids)  # , attention_mask=attention_mask)
    # <- Select the only sequence from the batch
    last_hidden_state = encoding.last_hidden_state

    # Average embeddings over multi-token exemplars
    embedding = []
    for item in item_indices:
        embedding += [torch.mean(last_hidden_state[0, item[0]+1:item[1]+1, :], dim=0)]

    # Calculate cosine distances between exemplars
    e1l = []
    for e1 in embedding:
        e2l = []
        for e2 in embedding:
            cosine_distance = 1 - F.cosine_similarity(e1, e2, dim=0)
            e2l += [float(cosine_distance)]
        e1l += [e2l]

    bert_distances = e1l

    return bert_distances
