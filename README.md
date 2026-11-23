![SemShift banner: a wall of English adverbs](docs/assets/banner.jpg)
<sub>Banner adapted from a photo by [Jason Leung](https://unsplash.com/@ninjason) on [Unsplash](https://unsplash.com/photos/a-black-and-white-photo-of-a-computer-screen-LiyazcC4xnQ) (Unsplash License).</sub>

# SemShift: *You shall know a word by the **corpus** it keeps*

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3776AB?logo=python&logoColor=fff)](https://www.nltk.org)
[![gensim](https://img.shields.io/badge/gensim-4.4-orange)](https://radimrehurek.com/gensim/)
[![networkx](https://img.shields.io/badge/networkx-3-2c7fb8)](https://networkx.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=fff)](https://scikit-learn.org)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matpato/SemShift/blob/main/notebooks/SemShift_biomedical.ipynb)

A hands-on tutorial on **lexical and vector semantics** for the NLP / PLN course (MEIC & MEIM, ISEL), Module 4.

We let a computer read **two very different worlds**, 20,000 abstracts of randomised controlled trials (**PubMed 20k RCT**)
and 1987 financial news (**Reuters-21578**), and learn word meanings from scratch. Then we ask both the same question:
*what does `delivery` mean?*

| word | PubMed RCT | Reuters 1987 |
|---|---|---|
| depression | anxiety, depressive, BDI-II | deteriorating, disruptions |
| delivery | cesarean, breech, deliveries | shipment, wheats, laydays |
| administration | intravenous, injection | Reagan, aides, Congress |
| confidence | interval, CI, odds, hazard | climate, fortunes |
| `MS` | multiple sclerosis, mass spectrometry | Morgan Stanley (ticker) |

The same story runs through **every learning outcome** of the module, and the whole notebook runs on a free Colab CPU
in **about 8–10 minutes**.

Website: <https://matpato.github.io/SemShift/> (from `docs/`).

---

## Tutorial overview

| § | Topic | Learning outcome |
|---|---|---|
| 1 | WordNet: the senses of *depression*, *delivery* and *administration*; synonyms, hypernyms, meronyms of the heart | LO1 lexical semantics |
| 2 | The company a word keeps: context counts of *delivery* and *pressure* in each corpus | LO2 distributional hypothesis |
| 3 | A TF-IDF search engine over 20,000 abstracts (*heart attack* vs. *myocardial infarction*); word–context counts vs. PPMI | LO3 sparse vectors, LO4 cosine |
| 4 | word2vec (skip-gram, negative sampling) trained on each corpus + pre-trained GloVe: the same word in three worlds; analogies | LO3 dense vectors, LO6 training |
| 5 | **Semantic-shift detector**: Procrustes alignment finds *confidence, hearing, terminal, branch* automatically | LO4 |
| 6 | Word sense disambiguation of *delivery* (birth, shipment, manner of speaking): MFS, domain MFS, Lesk, embeddings, on 26 hand-labelled sentences | LO5 WSD |
| 7 | Entity linking of the abbreviations `MS` and `PD`: corpus priors, dense context, NIL detection, live Wikidata candidates | LO5 entity linking |
| 8 | Limitations: antonyms (*positive* ≈ *negative*), new drugs missing from GloVe, one vector per word, bias | LO6 limitations |
| 9 | Graphs and metrics: a word graph of trial topics, graph-based sense induction, purity/NMI/ARI, precision/recall/F1, NIL | all |

Every step has a *predict before you run* prompt and a *discuss* box.

---

## Repository structure

| Folder / file | Description |
|---|---|
| `notebooks/SemShift_biomedical.ipynb` | The tutorial, Colab-ready, **saved with outputs** (readable on GitHub without running it) |
| `data/wsd_delivery_gold.json` | 26 occurrences of *delivery* (12 Reuters, 14 PubMed) hand-labelled with WordNet 3.0 senses |
| `data/el_biomed_kb.json`, `data/el_biomed_mentions.json` | Abbreviation KB (`MS`, `PD`, with Wikidata IDs) and 8 test mentions with gold links |
| `scripts/build_biomedical.py` | Regenerates the notebook from source |
| `slides/` | Beamer slides for the lecture (`oceansand` theme), their `.bib` file and figures (`slides/img_semantics_bio/`) |
| `docs/` | Tutorial web page (GitHub Pages) and the instructor guide |
| `requirements.txt` | Python dependencies |

---

## How to run

**Google Colab (recommended):** click the badge above, then *Runtime → Run all*. No GPU needed.

**Locally:**
```bash
pip install -r requirements.txt
jupyter lab notebooks/SemShift_biomedical.ipynb
```
The first run downloads PubMed 20k RCT (about 39 MB, from GitHub), the NLTK corpora (about 20 MB) and GloVe (128 MB).

| Step | Time (2-core CPU) |
|---|---|
| Download and clean corpora | about 30 s |
| word2vec PubMed / Reuters | about 2 min / 30 s |
| GloVe download | 30 s – 2 min |
| Graphs and metrics (§9) | about 30 s |
| **Total** | **about 4–10 min** |

---

## Data sources
* **PubMed 20k RCT** (Dernoncourt & Lee, 2017), downloaded from [GitHub](https://github.com/Franck-Dernoncourt/pubmed-rct).
* **Reuters-21578** (ApteMod split, via NLTK): 10,788 Reuters news documents from 1987.
* **WordNet 3.0** and the **Open Multilingual WordNet** (via NLTK).
* **GloVe** `glove-wiki-gigaword-100` (Pennington et al., 2014), via `gensim-data`.
* **Wikidata** identifiers for entity linking.

This tutorial is about **language**, not medicine: nothing in it is clinical advice.

---

## About these materials
This tutorial was prepared with the help of Claude, an AI assistant developed by Anthropic, and reviewed by the instructor.
Word2vec training with several threads is not fully deterministic, so your nearest neighbours may differ slightly
from the ones quoted in the text; the meanings are stable. If you find an error, please open an issue.

## License
MIT. Corpora and pre-trained vectors keep their original licences. Banner photo: Jason Leung, Unsplash License.
