from src.llm import generate_endpoint, FLUENCY_PROMPT
import networkx as nx
import random
import torch
import torch.nn.functional as F
import numpy as np


def normalize_probs(probs):
    """
    Normalize and sort a dict of probabilities
    """
    norm = sum(probs.values())
    norm_prob = {k: v / norm for k, v in probs.items()}
    norm_prob = dict(sorted(norm_prob.items(), key=lambda item: item[1], reverse=True))
    return norm_prob


def remove_dupliates(exemplar, probs):
    """
    Given a list of exemplars and probabilities, combine any duplicate entries
    and re-sort according to new probabilities
    """
    new_exemplar, new_probs = [], []
    for i, e in enumerate(exemplar):
        if e not in new_exemplar:
            new_exemplar += [e]
            new_probs += [probs[i]]
        else:
            new_probs[new_exemplar.index(e)] += probs[i]
    new_exemplar, new_probs = zip(*sorted(list(zip(new_exemplar, new_probs)), key=lambda x: x[1], reverse=True))
    return new_exemplar, new_probs


def get_next_node(graph, current_node):
    """
    Given a graph and node, return all valid next edges
    """
    valid_next_nodes = []
    for next_node in graph:
        if next_node == 'animal' or next_node == current_node or not graph.has_edge(current_node, next_node):
            continue
        valid_next_nodes += [next_node]
    return valid_next_nodes


def cue_switch_step(graph, current_node, beta_l, beta_g):
    """
    Return next node probabilities on a graph using the Hills cue switching model.

    Formalized by Avery et al., 2018:
    P(X_{n+1}|Q1, Q2, X_n) = (W(X_{n+1}, X_n)^beta_l * W(X_{n+1}, "animal")^beta_g) / (SUM_k=1^N W(X_{n+1}, X_k)^beta_l * W(X_{n+1}, "animal")^beta_g)
    """
    probs = {}
    for next_node in get_next_node(graph, current_node):
        # Get P(next node | current node)
        local_cue = graph[current_node][next_node]['weight']**beta_l

        # Get P(next node | "animal")
        global_cue = graph[next_node]['animal']['weight']**beta_g

        if current_node == 'animal': local_cue = 1

        probs[next_node] = local_cue * global_cue
    
    probs = normalize_probs(probs)
    return probs


def random_walk_step(graph, current_node, p):
    """
    Return next node probabilities on a graph using the Abbott random walk model.

    Formalized by Avery et al., 2018:
    P(X_{n+1}|Q1, Q2, X_n) = p * P(X_{n+1}|Q2) + (1 - p) * P(X_{n+1}|Q1, X_{n+1}, X_n)
    and
    P(a|Q) = W(a,Q) / SUM_k=1^N W(a_k, "animal")
    """
    probs = {}
    for next_node in get_next_node(graph, current_node):
        # Get P(next node | current node)
        local_cue = graph[current_node][next_node]['weight'] * p

        # Get P(next node | "animal")
        global_cue = graph[next_node]['animal']['weight'] * (1 - p)

        if current_node == 'animal': local_cue = 0

        probs[next_node] = local_cue + global_cue
    
    probs = normalize_probs(probs)
    return probs


def sub_category_switch_step(graph, current_node, category_map, transition_probs, beta_l=1, beta_g=5, beta_c=5):
    """
    Calculates a category switch with:
    - 1) A local cue
    - 2) A category cue accounting for subcategory transition probs
    - 3) A global cue
    """
    probs = {}
    for next_node in get_next_node(graph, current_node):
        # Get P(next node | current node)
        local_cue = graph[current_node][next_node]['weight']**beta_l

        # Get P(next node | "animal")
        global_cue = graph[next_node]['animal']['weight']**beta_g

        # Get P(next category | current category)
        if current_node == 'animal':
            category_cue = 1
        else:
            curr_cat, next_cat = category_map[current_node], category_map[next_node]
            common_cat = set(curr_cat).intersection(next_cat)
            if bool(common_cat):
                category_cue = max([transition_probs[c][c] for c in common_cat])
            else:
                category_cue = max([transition_probs[c1][c2] for c1 in curr_cat for c2 in next_cat])
            category_cue = category_cue**beta_c

        if current_node == 'animal': local_cue = 1
        if next_node == 'animal': category_cue = 1

        probs[next_node] = local_cue * global_cue * category_cue
    
    probs = normalize_probs(probs)
    return probs


def llm_step(graph, previous_seq, beta_l, beta_g, port, category_map=None):
    """
    Use a language model endpoint as a local cue to determine the next node.
    """
    probs = {}
    current_node = previous_seq[-1]

    # Generate next exemplars with probabilities
    exemplar, log_probs = generate_endpoint(previous_seq, prompt=FLUENCY_PROMPT, port=port)
    llm_probs = F.softmax(torch.tensor(log_probs, dtype=torch.float32), dim=0)
    exemplar, llm_probs = remove_dupliates(exemplar, llm_probs)

    # Ensure output probabilities are floats
    if isinstance(llm_probs, torch.Tensor):
        llm_probs = llm_probs.tolist()
    elif isinstance(llm_probs[0], torch.Tensor):
        llm_probs = [p.item() for p in llm_probs]

    # Only accept candidates which are defined by the categorization
    if category_map is not None:
        exemplar, llm_probs = [e for e in exemplar if e in category_map.keys()], [round(p, 4) for i, p in enumerate(llm_probs) if exemplar[i] in category_map.keys()]

    for next_node in get_next_node(graph, current_node):
        # Get P(next node | full sequence)
        if next_node in exemplar:
            local_cue = llm_probs[exemplar.index(next_node)]**beta_l
        else:
            local_cue = 1

        # Get P(next node | "animal")
        global_cue = graph[next_node]['animal']['weight']**beta_g

        if current_node == 'animal': local_cue = 1

        probs[next_node] = local_cue * global_cue
    
    probs = normalize_probs(probs)
    return probs


def dfs(graph, start_node='animal', max_length=40):
    """
    Perform a depth first traversal on a graph. Disagreements are decided by edge weights.
    """
    dfs_tree = nx.dfs_tree(graph, source=start_node)
    return [n for n in dfs_tree][:max_length]


def random_walk(graph, start_node='animal', max_length=40, max_iter=200, return_path=False):
    """
    Perform a random walk on a graph.
    """
    current_node = start_node
    walk, path = [current_node], [current_node]
    iter = 0

    while len(walk) < max_length and iter < max_iter:
        neighbors = list(graph.neighbors(current_node))
        if len(neighbors) > 0:
            next_node = random.choice(neighbors)
        else:
            next_node = start_node
        current_node = next_node

        path.append(next_node)
        if next_node not in walk:
            walk.append(next_node)
        iter += 1
    
    if return_path:
        return walk, path
    return walk


def random_walk_irt(path, exemplar):
    """Find the length of the first cycle in a path"""
    try:
        e0 = path.index(exemplar)
    except ValueError:
        return float('nan')
    
    try:
        e1 = path.index(exemplar, e0+1)
    except ValueError:
        return float('nan')

    return e1 - e0


def graph_traversal(graph, traversal_func, start_node='animal', traversal_method='max', max_length=40, max_iter=300, temp=None):
    """
    Generate a candidate list by traversing to the most likely candidate
    at each step, returning to the animal node is no candidates are found.

    Note: This algorithm explicitly does not allow duplicates.
    """
    current_node = start_node
    walk = [current_node]
    iter = 0

    while len(walk) < max_length and iter < max_iter:
        # print(f'{iter}/{max_iter}: {current_node}')
        iter += 1
        candidate_nodes = traversal_func(graph, walk, current_node)
        candidate_nodes = {k: v for k, v in candidate_nodes.items() if k not in walk}

        if len(candidate_nodes) == 0: 
            current_node = 'animal'
            continue

        if traversal_method == 'max':
            next_node = max(candidate_nodes, key=candidate_nodes.get)
        elif traversal_method == 'sample':
            cand_names, probs = list(candidate_nodes.keys()), list(candidate_nodes.values())
            if temp is not None:
                # Apply temperature and softmax to output probabilities
                probs = F.softmax(torch.asarray(probs) / temp, dim=-1)
            next_node = random.choices(cand_names, weights=probs)[0]

        current_node = next_node
        
        if next_node not in walk:
            walk += [next_node]
    
    if walk[0] == 'animal': walk = walk[1:]
    return walk

def beam_search(graph, traversal_func, start_node='animal', beam_width=10, max_length=40):
    """
    Implements beam search for a candidate traversal function on a graph.

    Unlike graph_traversal(), we do not allow traversing to previous nodes, but allow traversing to the 
    start_node if no other nodes are available, and then prune these entries before returning

    Note: We don't perform a log_softmax on the output probabilities.
    """
    # Create utils for converting graph outputs into tensors
    walk = None
    vocab_size, cand_names = len(graph), list(graph.nodes())
    cand_name_map, rev_cand_map = {c: i for i, c in enumerate(cand_names)}, {i: c for i, c in enumerate(cand_names)}
    indices = torch.tensor([cand_name_map[c] for c in cand_names])
    one_hot_encoded = F.one_hot(indices, num_classes=vocab_size)

    # Create a beam matrix of size (beam_width, 1)
    X = torch.zeros((beam_width, 1))

    # Pass through the model
    candidate_nodes = traversal_func(graph, walk, start_node)
    next_probs = torch.tensor([candidate_nodes[c] if c in candidate_nodes.keys() else 0 for c in cand_names], dtype=torch.float64)
    next_probs = torch.sum(one_hot_encoded * next_probs, dim=1)

    # Add the top-k single node candidates to the beam
    probabilities, idx = next_probs.topk(k=beam_width, axis=-1) # .squeeze().log_softmax(-1)
    X = torch.cat((X, idx[:, None]), axis=-1)
    
    for _ in range(max_length):
        # Run all candidate nodes through the model to get k*k next predictions
        next_probs = []
        frontier_nodes = [rev_cand_map[idx.item()] for idx in X[:, -1]] # Note, only provides most recent nodes
        for beam_idx, node in enumerate(frontier_nodes):
            candidate_nodes = traversal_func(graph, walk, node)
            entry_next_prob = torch.tensor([candidate_nodes[c] if c in candidate_nodes.keys() else 0 for c in cand_names], dtype=torch.float64)
            entry_next_prob[X[beam_idx].to(torch.int)] = 0 # If the entry has occured previously in the beam, don't traverse to that node
            if entry_next_prob.sum() == 0: entry_next_prob[cand_name_map[start_node]] = 1 # If no nodes are possible, return to the start node
            entry_next_prob = torch.sum(one_hot_encoded * entry_next_prob, dim=1)
            next_probs += [entry_next_prob]
        next_probs = torch.stack(next_probs, dim=0)

        # Add the next node probabilities to the beam probabilities
        probabilities = probabilities[:, None] + next_probs

        # Calculate top-k candidates of 2D probabilities, keep k highest probability beams and extend beam
        _, idx = probabilities.flatten().topk(k=beam_width, axis=-1)
        r, c = idx // probabilities.shape[1], idx % probabilities.shape[1]
        probabilities = probabilities[r, c]
        X = torch.cat((X[r], c[:, None]), axis=-1)

    # Remove the first entry and use the candidate map to return exemplars
    X = X[:, 1:]
    return [[rev_cand_map[c.item()] for c in run if c != cand_name_map[start_node]] for run in X], probabilities


def MC3(graph, traversal_func, start_node='animal', num_chains=10, max_length=40, partition=4):
    """
    Replication of MC3 from Zhu et al., 2018.

    The key innovation is running multiple Monte-Carlo Sampling chains at different temperatures, 
    where the low temperatures explore close nodes while the high temperature chains find new semantic
    patches. 

    Summary of changes:
    - Our initial position is the "animal node"
    - We are no longer sampling from a distribution function, we sample from the distribution of the
      current node to the next node

    Note:
    - Other algorithms prevent revisiting nodes. We can do that, or scrub repeated nodes out of the output
    - Also I need to get rid of the numpy from this function
    """
    walk = None
    vocab_size, cand_names = len(graph), list(graph.nodes())
    cand_name_map, rev_cand_map = {c: i for i, c in enumerate(cand_names)}, {i: c for i, c in enumerate(cand_names)}
    indices = torch.tensor([cand_name_map[c] for c in cand_names])
    one_hot_encoded = F.one_hot(indices, num_classes=vocab_size)

    # Initialize a range of temperatures
    beta = 1.0 / (1 + partition * np.arange(num_chains))
    
    # Markov chains initialization
    chain = np.zeros((max_length, num_chains))
    chain[0, :] = cand_name_map[start_node]        # [NEW] Initialize each chain with the start node

    # Simulation with MC3 algorithm
    for iter in range(1, max_length):
        # Sample new values for each chain, accepting at different temperatures
        for chain_idx in range(num_chains):
            curr_node = chain[iter - 1, chain_idx]
            
            # Add U(0, sigma) uniform noise corresponding to the mean posterior
            # new = curr_node + np.random.normal(0, sigma)

            # Recover forward probability P(x_{n+1} | x)
            candidate_nodes = traversal_func(graph, walk, rev_cand_map[curr_node])
            forward_probs = torch.tensor([candidate_nodes[c] if c in candidate_nodes.keys() else 0 for c in cand_names], dtype=torch.float64)
            forward_probs = torch.sum(one_hot_encoded * forward_probs, dim=1)

            # [NEW] Estimate a distribution of new exemplars given the current exemplar
            # Two ways of doing this:
            # (1) uniformly randomly pick a node
            # new_node = torch.randint(0, vocab_size, (1,)).item()
            # (2) sample a new node given prev. exemplar probability
            if forward_probs.sum() == 0: forward_probs[cand_name_map[start_node]] = 1 # If no nodes are possible, return to the start node
            new_node = torch.multinomial(forward_probs, 1).item()
            # (3) sample a new node given global probability
            # TODO

            forward_prob = forward_probs[new_node]

            # Recover backward probability P(x | x_{n+1})
            candidate_nodes = traversal_func(graph, walk, rev_cand_map[new_node])
            backward_probs = torch.tensor([candidate_nodes[c] if c in candidate_nodes.keys() else 0 for c in cand_names], dtype=torch.float64)
            backward_prob = torch.sum(one_hot_encoded * backward_probs, dim=1)[int(curr_node)]

            # Acceptance probability newly sampled value at a random chance judgement, determined by temperature
            # A = min(1, (F(new) ** beta[chain_idx]) / (F(curr_node) ** beta[chain_idx]))

            # [NEW] Calculate the acceptance ratio as P(new -> curr_node) / P(curr_node -> new)
            A = min(1, (forward_prob ** beta[chain_idx]) / (backward_prob ** beta[chain_idx]))
            if random.random() < A:
                chain[iter, chain_idx] = new_node
            else:
                chain[iter, chain_idx] = curr_node

        # Compare the probabilities at the end of the chains and perform psuedo-random swapping
        if num_chains > 1:
            sChain = np.random.permutation(num_chains)
            for k in range(0, len(sChain)//2):
                m, n = sChain[2 * k], sChain[2 * k + 1]
                # top = (F(chain[iter, m]) ** beta[n]) * (F(chain[iter, n]) ** beta[m])
                # bottom = (F(chain[iter, m]) ** beta[m]) * (F(chain[iter, n]) ** beta[n])

                # [NEW] We estimate the probability of being within the state as P(new_node|curr_node) for both chains
                prev_node, curr_node = chain[iter, m-1], chain[iter, m]
                candidate_nodes = traversal_func(graph, walk, rev_cand_map[prev_node])
                forward_probs = torch.tensor([candidate_nodes[c] if c in candidate_nodes.keys() else 0 for c in cand_names], dtype=torch.float64)
                forward_prob_top = torch.sum(one_hot_encoded * forward_probs, dim=1)[int(curr_node)]

                prev_node, curr_node = chain[iter, n-1], chain[iter, n]
                candidate_nodes = traversal_func(graph, walk, rev_cand_map[prev_node])
                forward_probs = torch.tensor([candidate_nodes[c] if c in candidate_nodes.keys() else 0 for c in cand_names], dtype=torch.float64)
                forward_prob_bottom = torch.sum(one_hot_encoded * forward_probs, dim=1)[int(curr_node)]

                # Calculate acceptance probability of swapping MCMC chains
                top = (forward_prob_top ** beta[n]) * (forward_prob_bottom ** beta[m])
                bottom = (forward_prob_top ** beta[m]) * (forward_prob_bottom ** beta[n])
                A_swap = min(1, top / bottom)
                if random.random() < A_swap:
                    temp = chain[iter, m]
                    chain[iter, m] = chain[iter, n]
                    chain[iter, n] = temp

    # Convert candidate map back to exemplars
    return [[rev_cand_map[c.item()] for c in run if c != cand_name_map[start_node]] for run in chain.T]
