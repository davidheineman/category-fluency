import numpy as np
import random


def greedy_category_switches(input_categories, category_probs=None, exemplars=None):
    assignments = []
    for i, cat in enumerate(input_categories):
        if i != 0 and assignments[i-1] in cat:
            assignments += [assignments[i-1]]
        elif i > 1 and assignments[i-2] in cat:
            assignments += [assignments[i-2]]
        elif i > 2 and assignments[i-3] in cat:
            assignments += [assignments[i-3]]
        else:
            if category_probs is not None:
                assignments += [max(category_probs[exemplars[i]], key=lambda x: x[1])[0]]
            else:
                assignments += [random.choice(cat)]
    
    return assignments
        

def min_category_switches(input_categories, category_probs=None, exemplars=None, foreshadow_window_size=None):
    """
    Given a list of lists of categories, return a combination which minimizes the number
    of category switches.

    Implemented with a DP approach, where the DP array is O(# exemplars * # unique categories)
    """
    all_categories = set([i for j in input_categories for i in j])
    category_map = {cat: i for i, cat in enumerate(set(all_categories))}
    reverse_category_map = {i: cat for i, cat in enumerate(set(all_categories))}

    n, m = len(input_categories), len(category_map.keys())
    dp = [[float('inf') for _ in range(m)] for _ in range(n)]

    for category in input_categories[0]:
        cat_idx = category_map[category]
        dp[0][cat_idx] = 0

    for i in range(1, n):
        min_last_entry = min(dp[i-1]) + 1
        for category in input_categories[i]:
            cat_idx = category_map[category]

            # If the category in the last column is not inf, then use it in the next column
            if dp[i-1][cat_idx] != float('inf'):
                dp[i][cat_idx] = min(dp[i-1][cat_idx], min_last_entry)
            else:
                # Get the minimum of all previous entires
                dp[i][cat_idx] = min_last_entry

    # Recover category assignments by backtracking through DP array
    assignments = [None] * n
    
    for i in range(n-1, -1, -1):
        min_indices = [j for j, x in enumerate(dp[i]) if x == min(dp[i])]
        assignments[i] = reverse_category_map[min_indices[0]]

        if i == n-1:
            continue
        
        # for min_idx in min_indices:
        #     category = reverse_category_map[min_idx]
        #     if category == assignments[i+1]:
        #         assignments[i] = category
        #         continue

        # Random selection
        assignments[i] = reverse_category_map[random.choice(min_indices)]

        # Foreshadowing
        if foreshadow_window_size is not None:
            for min_idx in min_indices:
                category = reverse_category_map[min_idx]
                for j in range(foreshadow_window_size if i + foreshadow_window_size <= n-1 else i-(n-1)):
                    if category == assignments[i+j]:
                        assignments[i] = category
                        continue

        # Category frequency
        if category_probs is not None:
            max_freq = -1
            for min_idx in min_indices:
                category = reverse_category_map[min_idx]
                if category in [c[0] for c in category_probs]:
                    cat_freq = [c for c in category_probs[exemplars[i]] if c[0] == category][0][1]
                    if cat_freq > max_freq:
                        max_freq = cat_freq
                        assignments[i] = category

    return assignments


def category_freq(category_mapping, data):
    exemplar_freq = {}
    for run in data:
        category_choices = [category_mapping[r] for r in run['response'] if r in category_mapping]
        categories = min_category_switches(category_choices)

        for exemplar, category in zip(run['response'], categories):
            if exemplar not in exemplar_freq.keys():
                exemplar_freq[exemplar] = []
            exemplar_freq[exemplar] += [category]
    return exemplar_freq


def category_ratios(category_mapping, data):
    exemplar_freq = category_freq(category_mapping, data)
    for exemplar in exemplar_freq.keys():
        e_counts = exemplar_freq[exemplar]
        exemplar_freq[exemplar] = [(i, e_counts.count(i) / len(e_counts)) for i in set(e_counts)]
    return exemplar_freq


def adjudicate_categorization(category_mapping, data):
    """
    Given a mapping of exemplar to many categories, use human data to
    determine the most common category given human responeses. Useful
    for comparing categorization quality using clustering.
    """
    exemplar_freq = category_freq(category_mapping, data)

    for exemplar in exemplar_freq.keys():
        exemplar_freq[exemplar] = max(set(exemplar_freq[exemplar]), key=exemplar_freq[exemplar].count)
    
    return exemplar_freq


def get_mean_heuristic(hills, heuristic_col, switch_col='switch_adj'):
    """
    Given the hills data and a column, calculate the average of that column centered
    around a category switch, either defined by the hills data or a category mapping.

    Ignores all float('nan') values
    """
    all_switch_vals = []
    for i, run in enumerate(hills): 
        if heuristic_col not in run.keys():
            # raise KeyError(f'Heuristic {heuristic_col} not found in run. Seeing keys: {run.keys()}')
            continue
        
        switches, val = run[switch_col], run[heuristic_col]

        assert len(switches) == len(val) == len(run['response']), f'Run {i} does not have consistent lengths!'

        # Set any inf values to nan
        val = np.asarray(val)
        if np.any(np.isinf(val)):
            val[val == np.inf] = np.nan

        switch_vals = [
            # [val[i-2], val[i-1], val[i], val[i+1], val[i+2]]
            [val[i-3], val[i-2], val[i-1], val[i], val[i+1]]
            for i, c in enumerate(switches)
            if i-3 >= 0 and i+2 < len(switches)
            and switches[i] == True
        ]
                
        if len(switch_vals) == 0: continue

        all_switch_vals += [np.nanmean(switch_vals, axis=0) / np.nanmean(val)]
    return np.nanmean(all_switch_vals, axis=0)
    

def get_category_transitions(hills):
    """
    Extract a transition matrix from the hills data
    """
    transitions = []
    for run in hills:
        transitions += [[run['category'][i-1], run['category'][i]] for i, c in enumerate(run['category'][1:])]
    
    states = list(set([i for j in transitions for i in j]))
    state_to_index = {state: i for i, state in enumerate(states)}

    transition_matrix = np.zeros((len(states), len(states)))
    for entry in transitions:
        transition_matrix[state_to_index[entry[0]], state_to_index[entry[1]]] += 1

    # return states, transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    return {s: {s2: float(t2) for s2, t2 in zip(states, t)} for s, t in zip(states, transition_matrix / transition_matrix.sum(axis=1, keepdims=True))}