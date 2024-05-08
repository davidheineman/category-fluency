# Category Fluency
Code for "Towards a path dependent account of category fluency" at CogSci 2024. Existing models are implemented in [`src`](/src/), with analysis and figures in the [`analysis`](/analysis/) notebooks.

**A quick-start Google Collab Notebook is coming soon!**

## Data

Hills et al., 2012 data was taken from: [supp.apa.org/psycarticles/supplemental/a0027373/a0027373_supp.html](https://supp.apa.org/psycarticles/supplemental/a0027373/a0027373_supp.html).

To download all data automatically, please use:

```sh

```

## Setup

#### Notebooks Server

For analysis notebooks, simply install the dependencies and run the notebooks:

```
pip install -r requirements.txt
```

#### LLM Server

For experiments using LLaMA 7B, we run the language model as an endpoint. To run the LLM server:

```sh
pip install -r requirements.txt
cd cli
python endpoint.py
```

This will create an API endpoint, which you can connect to in the `2_llm.ipynb` notebook to run experiments.