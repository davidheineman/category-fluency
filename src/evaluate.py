import statistics, itertools
from nltk.translate.bleu_score import sentence_bleu
from .switch import min_category_switches


def sequence_bleu(gold, pred):
    scores = []
    for i, r in enumerate(pred):
        scores += [sentence_bleu(gold, r, weights=(0.5, 0.5, 0.5))]
    return statistics.mean(scores)


def round_dict(results):
    return {k: ((round(v, 6) if v > 1e-6 else 0) if not isinstance(v, dict) else v) for k, v in results.items()}


def evaluate_results(pred, gold, cateogry_mapping):
    results = {}

    if not isinstance(pred[0], list): pred = [pred]

    results['exemplar_bleu'] = sequence_bleu(gold, pred)

    pred_categories = []
    for p_seq in pred:
        pred_categories += [min_category_switches([cateogry_mapping[p] for p in p_seq])]
    
    gold_categories = []
    for g_seq in gold:
        gold_categories += [min_category_switches([cateogry_mapping[g] for g in g_seq])]

    results['intra_category_bleu'] = sequence_bleu(gold_categories, pred_categories)

    intra_category_results = {}
    valid_categories = set([i for j in cateogry_mapping.values() for i in j])
    for category in valid_categories:
        pred_cat_all = []
        for i in range(len(pred_categories)):
            pred_cat_all += [[pred[i][j] for j in range(0, len(pred[i])) if pred_categories[i][j] == category]]

        gold_cat_all = []
        for i in range(len(gold_categories)):
            gold_cat_all += [[gold[i][j] for j in range(0, len(gold[i])) if gold_categories[i][j] == category]]

        pred_cat_all = [p for p in pred_cat_all if len(p) > 0]
        gold_cat_all = [p for p in gold_cat_all if len(p) > 0]

        if len(pred_cat_all) > 0 and len(gold_cat_all) > 0:
            bleu = sequence_bleu(gold_cat_all, pred_cat_all)
        else:
            bleu = 0
        
        intra_category_results[category] = bleu
    results['inter_category_bleu'] = round_dict(intra_category_results)

    run_lengths = []
    for p_seq in pred_categories:
        run_lengths += [len(list(v)) for k, v in itertools.groupby(p_seq)]
    results['average_run_length'] = statistics.mean(run_lengths)

    switches = []
    for p_seq in pred_categories:
        switches += ([False] + [p_seq[i-1] != p_seq[i] for i, c in enumerate(p_seq[1:])])
    results['percent_switches'] = statistics.mean(switches)

    results = round_dict(results)

    return results