![SemShift banner: a wall of English adverbs](docs/assets/banner.jpg)
<sub>Banner adapted from a photo by [Jason Leung](https://unsplash.com/@ninjason) on [Unsplash](https://unsplash.com/photos/a-black-and-white-photo-of-a-computer-screen-LiyazcC4xnQ) (Unsplash License).</sub>

# SemShift: *You shall know a word by the **corpus** it keeps*

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3776AB?logo=python&logoColor=fff)](https://www.nltk.org)
[![gensim](https://img.shields.io/badge/gensim-4.4-orange)](https://radimrehurek.com/gensim/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=fff)](https://scikit-learn.org)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/matpato/SemShift/blob/main/notebooks/SemShift_tutorial.ipynb)

A hands-on tutorial on **lexical and vector semantics** for the NLP / PLN course (MEIC & MEIM, ISEL), Module 4.

We let a computer read **two very different worlds**, 1987 financial news (**Reuters**) and three centuries of
literature (**Project Gutenberg**: Austen, Shakespeare, the King James Bible, Melville…), and learn word meanings
from scratch. Then we ask both the same question: *what does `bank` mean?*

| word | 🏦 Reuters 1987 | 📜 Gutenberg |
|---|---|---|
| bank | central, england, banks, citibank | pool, meadow, footpath, cafe |
| light | sweet, bonny, crudes *(Bonny light, a crude oil grade)* | shineth, stars, darkness |
| mine | gold, copper, ounces | thine, my *(a pronoun!)* |
| house | senate, subcommittee, speaker | chamber, tabernacle, tent |
| tender | offer, tenders *(a tender offer)* | giver, glow, wholesome |

The same story runs through **every learning outcome** of the module, and the whole notebook runs on a free Colab CPU in **≈ 8–10 minutes**.

---

## 📚 Tutorial overview

| § | Topic | Learning outcome |
|---|---|---|
| 1 | WordNet: senses, synonyms, hypernyms, hyponyms, meronyms, antonyms; *banco* in Portuguese | LO1 lexical semantics |
| 2 | Keyword-in-context: the company `bank` keeps in each world | LO2 distributional hypothesis |
| 3 | TF-IDF book vectors (authors cluster by themselves) and PPMI word vectors (counts vs. PPMI) | LO3 sparse vectors, LO4 cosine |
| 4 | word2vec (skip-gram, negative sampling) trained on each corpus + pre-trained GloVe: the same word in three worlds | LO3 dense vectors, LO6 training |
| 5 | ⭐ **Semantic-shift detector**: Procrustes alignment finds *tender, chase, fed, suit, common* automatically | LO4 |
| 6 | Word sense disambiguation: MFS, *domain* MFS, Lesk, embeddings, on 24 hand-labelled `bank` tokens | LO5 WSD |
| 7 | Entity linking to Wikidata: *Jordan* (country, river, basketball player), *Shell*, NIL detection, live Wikidata candidates | LO5 entity linking |
| 8 | Limitations: one vector per word, antonyms, OOV, **gender bias in 19th-century novels vs. GloVe** | LO6 limitations |
| 9 | 🇵🇹 Bonus: Machado de Assis vs. *Folha de S. Paulo* (1994): *rede*, *pena*, *real* | all |

Every step has a 🔮 *predict before you run* prompt and a 💬 *discuss* box.

---

## 📁 Repository structure

| Folder / file | Description |
|---|---|
| `notebooks/SemShift_tutorial.ipynb` | The tutorial, Colab-ready, **saved with outputs** (readable on GitHub without running it) |
| `data/wsd_bank_gold.json` | 24 occurrences of *bank* (12 Reuters, 12 Gutenberg) hand-labelled with WordNet 3.0 senses |
| `data/el_kb.json`, `data/el_mentions.json` | Tiny entity-linking KB (Wikidata IDs) and 9 test mentions with gold links |
| `scripts/build_notebook.py` | Regenerates the notebook from source |
| `docs/` | Tutorial web page (GitHub Pages) and the instructor guide |
| `requirements.txt` | Python dependencies |

---

## ▶️ How to run

**Google Colab (recommended):** click the badge above → *Runtime → Run all*. No GPU needed.

**Locally:**
```bash
pip install -r requirements.txt
jupyter lab notebooks/SemShift_tutorial.ipynb
```
The first run downloads the NLTK corpora (≈ 20 MB) and GloVe (128 MB).

| Step | Time (2-core CPU) |
|---|---|
| Load and clean corpora | ≈ 20 s |
| PPMI matrices (2 × 10k × 10k) | ≈ 5 s |
| word2vec Reuters / Gutenberg | ≈ 35 s / ≈ 55 s |
| GloVe download | ≈ 30 s – 2 min |
| Portuguese bonus (2 models) | ≈ 1.5 min |
| **Total** | **≈ 4–8 min** |

---

## 🔗 Data sources
* **Reuters-21578** (ApteMod split, via NLTK): 10,788 Reuters news documents from 1987.
* **Project Gutenberg selection** (via NLTK): 18 public-domain texts.
* **WordNet 3.0** and the **Open Multilingual WordNet** (via NLTK).
* **GloVe** `glove-wiki-gigaword-100` (Pennington et al., 2014), via `gensim-data`.
* **Machado de Assis** complete works and **Mac-Morpho** (*Folha de S. Paulo*, 1994), via NLTK.
* **Wikidata** identifiers for entity linking.

---

## 🤝 About these materials
This tutorial was prepared with the help of Claude, an AI assistant developed by Anthropic, and reviewed by the instructor.
Word2vec training with several threads is not fully deterministic, so your nearest neighbours may differ slightly
from the ones quoted in the text; the meanings are stable. If you find an error, please open an issue.

## 📜 License
MIT. Corpora and pre-trained vectors keep their original licences. Banner photo: Jason Leung, Unsplash License.
