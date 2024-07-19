<div align="center">
    <h1>Category Fluency</h1>

[**Quick Start Demo**](https://colab.research.google.com/drive/1C2tNUf0ao1hDgmWNZ_6ySYD2T5NefSwp?usp=sharing)  | [**Download Data**](/data) | [**View Paper**](https://arxiv.org/abs/2405.06714)
</div>

Code for *Towards a path dependent account of category fluency* at CogSci 2024. Category fluency models are implemented in [`/src`](/src/), with analysis and figures in the [`/analysis`](/analysis/) notebooks.

**A quick-start Google Collab Notebook [is available here](https://colab.research.google.com/drive/1C2tNUf0ao1hDgmWNZ_6ySYD2T5NefSwp?usp=sharing).** This notebook allows replicating the LLM category fluency experiments without the need to set up the entire code repo.

## Data

All data is included in [`/data`](/data/). We pre-formatted data into .csv files.

For the original sources: 
- The Hills et al., 2012 human experimental data was taken from: [supp.apa.org/psycarticles/supplemental/a0027373/a0027373_supp.html](https://supp.apa.org/psycarticles/supplemental/a0027373/a0027373_supp.html). 
- The Abbott et al., 2015 free association data was taken from Nelson et al., 2004 available at [https://link.springer.com/article/10.3758/BF03195588](https://link.springer.com/article/10.3758/BF03195588#SecESM1)

## Setup

#### Notebooks Server

For analysis notebooks, simply install the dependencies and run the notebooks:

```sh
pip install -r requirements.txt

# Download GloVe embeddings
curl -O https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip glove.6B
mv glove.6B.300d.txt data/glove.6B.300d.txt
rm glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt glove.6B.zip
```

#### LLM Server

For experiments using LLaMA 7B, we run the language model as an endpoint. To run the LLM server:

```sh
pip install -r requirements.txt
cd cli
python endpoint.py
```

This will create an API endpoint, which you can connect to in the `2_llm.ipynb` notebook to run experiments.