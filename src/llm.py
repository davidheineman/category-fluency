import datetime, os, requests, json
from vllm import LLM, SamplingParams

# Seperator token between exemplars
SEP = '\n'
FLUENCY_PROMPT = f"""List animals in whatever order comes first to mind, begin with the most animal-like example. Do not repeat the animals and only list the animals. Your list should be seperated by newlines."""
N_EXEMPLARS = 30

DEFAULT_KWARGS = {
    'use_beam_search': True,
    'num_beams': N_EXEMPLARS,
    'num_return_sequences': N_EXEMPLARS,
    'early_stopping': True,
    'stop': SEP,
    'max_new_tokens': 5,
    'logprobs': N_EXEMPLARS,
    'output_scores': True,
    'return_dict_in_generate': False, 
    'do_sample': False,
    'top_p': None,
    'temperature': None,
    'epsilon_cutoff': None,
}

VLLM_DIR = str(os.environ.get("HF_HOME"))
DEVICES = [int(d) for d in os.environ.get("CUDA_VISIBLE_DEVICES", '0').split(',')]

MODEL_ENDPOINT = 'http://localhost:{port}/generate'


def init_llm(model_name):
    """
    Note, rather than use torchrun to start multiple endpoints in parallel and distribute across them,
    we use 1 endpoint which manages GPUs within the vLLM abstraction.
    """
    global MODEL, DEVICES

    if DEVICES == -1:
        raise RuntimeError(f'Ensure CUDA_VISIBLE_DEVICES is set! Seeing {DEVICES}')

    print(f'Loading model: {model_name} to devices {DEVICES}...')

    MODEL = LLM(
        model=model_name, 
        tensor_parallel_size=len(DEVICES),
        max_model_len=4096,
        download_dir=VLLM_DIR
    )


def generate_exemplar(previous_exemplars: list, prompt: str=None):
    text = f'{prompt}\n{SEP.join(previous_exemplars)}\n'
    if isinstance(text, str): text = [text]
    return model_generate(text)


def model_generate(input_text):
    start_time = datetime.datetime.now()

    if not isinstance(input_text, list): input_text = [input_text]

    print(f'Generating {len(input_text)} examples on {DEVICES} with params {params}')
    
    # See: https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    params = SamplingParams(**DEFAULT_KWARGS)

    generation = MODEL.generate(
        prompts=input_text, 
        sampling_params=params,
        use_tqdm=True
    )

    # Measure generation time
    duration = (datetime.datetime.now() - start_time).total_seconds()
    gen_length = sum(sum(len(out.token_ids) for out in o.outputs) for o in generation)
    print(f"Generated {gen_length} tokens in {duration:.2f}s at {gen_length/duration:.2f} tok/s on {DEVICES}.")

    # Flatten nested outputs, helpful for beam search
    generation_text = [i for j in [[o.text for o in out.outputs] for out in generation] for i in j]

    seq_prob = [o.cumulative_logprob for o in generation[0].outputs]

    return generation_text, seq_prob


def generate_endpoint(previous_exemplars, prompt=FLUENCY_PROMPT, port=8474):
    data = { 'previous_exemplars': previous_exemplars, 'prompt': prompt }

    try:
        response = requests.post(MODEL_ENDPOINT.format(port=port), json=data)
        response_json = response.json()
    except json.JSONDecodeError:
        raise RuntimeError(f"Llama endpoint failed to respond. Returned: {response}")

    return response_json['generation']