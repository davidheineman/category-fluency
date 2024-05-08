import sys, os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(ROOT_DIR))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

from flask import Flask, request, jsonify
from llm import init_llm, generate_exemplar


MODEL_NAME = str(os.environ.get("MODEL_NAME", 'meta-llama/Llama-2-7b-chat-hf'))
MODEL_PORT = int(os.environ.get("MODEL_PORT", 8200))


app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    generation = generate_exemplar(**data)
    return jsonify({'generation': generation})


if __name__ == '__main__':
    init_llm(MODEL_NAME)

    # Sanity check
    exemplars = generate_exemplar(
        previous_exemplars=[
            'cat', 'dog', 'mouse', 'rat', 'giraffe', 'lion', 'zebra', 'rhino', 
            'ostrich', 'meerkat', 'tiger', 'bear', 'bat'
        ], 
        prompt="List unique animals, separated by newlines. Do not repeat the animals."
    )
    print(exemplars)

    app.run(host='0.0.0.0', port=MODEL_PORT)
