import csv
import numpy as np
from tqdm import tqdm


REBA_CATEGORY_COLORS = {
    'Arctic/Far North Animals': '#25a6b8',
    'Arctic/Far North Animals': '#25a6b8',
    'Australian Animals': '#FC887F',
    'Australian Animals': '#FC887F',
    'Beasts of Burden': '#0070C0',
    'Birds': '#FFD966',
    'Canines': '#FAD565',
    'Farm Animals': '#FFF2CC',
    'Felines': '#DDEBF7',
    'Fish': '#FCA7FC',
    'Insects': '#FFD4B8',
    'Jungle Animals': '#bf6fed',
    'Non-Jungle Animals': '#b87b25',
    'North American': '#93D050',
    'Pets': '#02B0F0',
    'Primates': '#bf6fed',
    'Reptiles/Amphibians': '#D9C7FF',
    'Rodent Like': '#b84a25'
}

REBA_CATEGORY_ABBR = {
    'Arctic/Far North Animals': 'arctic', 'Arctic/Far North Animals': 'arctic',
    'Australian Animals': 'australian', 'Australian Animals': 'australian',
    'Beasts of Burden': 'beasts', 'Birds': 'birds', 'Canines': 'canines', 'Farm Animals': 'farm',
    'Felines': 'felines', 'Fish': 'fish', 'Insects': 'insects', 'Jungle Animals': 'jungle',
    'Non-Jungle Animals': 'nonjungle', 'North American': 'north', 'Pets': 'pets', 'Primates': 'primates',
    'Reptiles/Amphibians': 'reptiles', 'Rodent Like': 'rodent'
}

TROYER_CATEGORY_COLORS = {
    'African animals': '#EDE1CF', #
    'Animals used for fur': '#20B2AA', #
    'Arctic/Far North Animals': '#25a6b8',
    'Australian Animals': '#FC887F',
    'Beasts of burden': '#0070C0',
    'Birds': '#FFD966',
    'Bovine': '#87CEEB', #
    'Canine': '#FAD565',
    'Deers': '#FF6347', #
    'Farm animals': '#FFF2CC',
    'Feline': '#DDEBF7',
    'Fish': '#FCA7FC',
    'Insectivores': '#D2691E', #
    'Insects': '#FFD4B8',
    'North American animals': '#93D050',
    'Pets': '#02B0F0',
    'Primates': '#bf6fed',
    'Rabbits': '#00FF7F', # 
    'Reptiles/Amphibians': '#D9C7FF',
    'Rodents': '#b84a25',
    'Water animals': '#FDD2FD', # 
    'Weasels': '#40E0D0' #
}


def load_glove_model(file_path):
    print("Loading GloVe...")
    glove_model = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def load_categories(path):
    with open(path, 'r', encoding='utf-8') as f:
        categories = [r for r in csv.reader(f)]
    category_map = {}
    for c in categories:
        category, exemplars = c[0], c[1:]
        for exemplar in exemplars:
            exemplar = exemplar.lower().replace(' ', '')
            if exemplar not in list(category_map.keys()):
                category_map[exemplar] = []
            category_map[exemplar] += [category]
    return category_map
    
    
def load_hills(path):
    hills = []
    with open(path, 'r', encoding='utf-8') as f:
        data = [r for r in csv.reader(f)]
    for entry in data[1:]:
        hills += [{
            'participant': int(entry[0]),
            'response': entry[1],
            'irt': float(entry[4]),
            'category': entry[3],
            'valid_categories': entry[2].split(','),
        }]
    return hills


def load_nelson(paths, valid_exemplars):
    data = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            data += [r for r in csv.reader(f)]

    cues = set([e[0] for e in data[3:]])
    exemplars = [e.upper() for e in valid_exemplars]

    nelson = {}
    for cue in cues:
        if cue not in exemplars:
            continue
        targets = [e[1] for e in data[3:] if e[0] == cue]
        nelson[cue.lower()] = list(set([t.lower() for t in targets if t.lower() in valid_exemplars]))
    
    return nelson


def load_word_frequency(path):
    with open(path, 'r', encoding='utf-8') as f:
        word_freq = {i[0]: float(i[1]) for i in [r for r in csv.reader(f)]}
    return word_freq


def preprocess_hills(hills, glove_model, category_mapping):
    """
    Group Hills data by runs and add columns for switching and semantic similarity
    """
    new_hills = []
    participants = list(set([e['participant'] for e in hills]))
    for p in participants:
        run = [e for e in hills if e['participant'] == p]
        new_hills += [{
            'participant': p,
            'response': [r['response'] for r in run],
            'irt': [r['irt'] for r in run],
            'category': [r['category'] for r in run],
            'valid_categories': [r['valid_categories'] for r in run],
        }]
    hills = new_hills

    from .similarity import prev_seq_similarity
    from .switch import min_category_switches

    for i, run in enumerate(hills):
        hills[i]['mean_irt'] = np.mean(run['irt'])
        hills[i]['glove'] = np.asarray(prev_seq_similarity(run['response'], glove_model))

        # Add switches from manual cateogry assignments
        hills[i]['switch_adj'] = [False] + [run['category'][i-1] != run['category'][i] for i, c in enumerate(run['category'][1:])]

        # Add switches from ajudicated categories
        categories = min_category_switches([category_mapping[r] for r in run['response']])
        hills[i]['switch_reba'] = [False] + [categories[i-1] != categories[i] for i, c in enumerate(categories[1:])]
        # hills[i]['switch_troyer'] = [False] + [run['category'][i-1] != run['category'][i] for i, c in enumerate(run['category'][1:])]
    
    return hills
