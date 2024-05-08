import os, sys, json
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR))

from src.data import load_categories, load_hills, load_glove_model, load_word_frequency, preprocess_hills
from src.tree import create_animal_graph
from src.search import llm_step

HILLS_DATA_PATH = '../data/hills.csv'
REBA_CATEGORIES_PATH = '../data/reba_categories.csv'
WORD_FREQ_PATH = '../data/word_frequency.csv'
GLOVE_PATH = '../data/glove/glove.6B.300d.txt'

MODEL_PORT = int(os.environ.get("MODEL_PORT", 8200))
HILLS_OUTPUT_PATH = 'out/hills.json'

def save_dict(data, filename):
    data_copy = data.copy()
    for i, _ in enumerate(data_copy):
        if "glove" in data_copy[i]:
            del data_copy[i]["glove"]
    with open(filename, 'w') as f:
        json.dump(data_copy, f, indent=4)

def main():
    """
    Given a language model endpoint, pre-computes LLM probabilities. Great for running
    as a batch job to perform data analysis over later-on.

    Usage: nohup python entropy.py > log/entropy.log &
    """
    hills_csv = load_hills(HILLS_DATA_PATH)
    reba_categories = load_categories(REBA_CATEGORIES_PATH)
    word_freq = load_word_frequency(WORD_FREQ_PATH)
    glove_model = load_glove_model(GLOVE_PATH)

    hills = preprocess_hills(hills_csv, glove_model, reba_categories)

    EPSILON = 0.4
    G = create_animal_graph(reba_categories.keys(), glove_model, word_freq, EPSILON)

    BETA_L, BETA_G = 1, 0

    for i, run in tqdm(enumerate(hills)):
        cue_probs = [None]
        for j, r in enumerate(run['response'][1:]):
            prefix = run['response'][:j+1]

            try:
                next_node = llm_step(G, prefix, BETA_L, BETA_G, MODEL_PORT)
                cue_probs += [next_node]
                if len(next_node) > 0:
                    print(f"{prefix} -> {list(next_node.keys())[0]} (with sum {sum(next_node.values()):.2f})")
            except Exception as e:
                cue_probs += [None]
                print(f'Failed on {prefix}, received error "{e}". Skipping...')
                continue
            
        hills[i]['llm_cue_probs'] = cue_probs
        save_dict(hills, HILLS_OUTPUT_PATH)

if __name__ == '__main__':
    main()