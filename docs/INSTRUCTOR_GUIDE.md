# Instructor guide: SemShift practical lecture (about 90 min)

Notebook: `notebooks/SemShift_biomedical.ipynb` · Slides: `slides/PLN_04_Semantics_Biomedical.tex` · Website: `docs/index.html`

## Before class (10 min, the day before or that morning)
1. Open the notebook in Colab and run **§0 Setup** plus the training cell of §4 once, to check that Colab, NLTK, gensim and the PubMed download still work.
2. Keep the GitHub page of the notebook open in a second tab. It is saved **with outputs**, so it is the fallback if Colab is slow or the Wi-Fi fails.
3. At the start of class, ask the students to open the Colab link and run **§0 to §4** straight away (about 3 min). By the time you reach §4, their models are ready.

## Lesson plan

| Time | Section | What to show live | The "aha" to land |
|---|---|---|---|
| 0–5 | Warm-up | Three *delivery* sentences: wheat, cesarean, aerosol vaccine | One word, three meanings |
| 5–15 | §1 WordNet | *depression* senses; meronyms of the heart; `depressive_disorder` vs. economic depression | WordNet is general; UMLS/MeSH/SNOMED are the clinical inventories |
| 15–22 | §2 Contexts | Context counts of *delivery* and *pressure* in each corpus | Meaning is visible in the neighbours, before any maths |
| 22–35 | §3 Sparse | TF-IDF search: "myocardial infarction" vs. "heart attack"; counts vs. PPMI | Exact words only: synonyms are invisible to sparse vectors |
| 35–48 | §4 Dense | The three-worlds table; analogies: *knee : arthroplasty :: hip* gives **hip-hop** (GloVe) vs. **THA** (PubMed); *aspirin : stroke :: statin* gives golf terms in GloVe | The dominant sense wins; domain matters |
| 48–56 | §5 Shift | *confidence* (interval), *intervention*, *hearing*, *controlled*, *branch* | Polysemy found without labels |
| 56–66 | §6 WSD | *delivery*: MFS 0.43 on PubMed; general GloVe beats PubMed word2vec | Domain vectors know the domain, not the dictionary |
| 66–74 | §7 EL | `MS` = multiple sclerosis / mass spectrometry / Morgan Stanley; corpus priors; NIL; live Wikidata cell | WSD vs. EL; priors are domain-dependent; NIL |
| 74–80 | §8 Limits | *positive*/*negative* cos about 0.9; drugs missing from GloVe; bias charts | Antonyms and OOV are clinical risks; association is not bias by itself |
| 80–88 | §9 Graphs | Word graph of trial topics (Q about 0.77); occurrence graph of *delivery* and its induced senses; classification reports | Senses can be *discovered*; accuracy hides failing classes (MFS: accuracy 0.65, macro F1 0.26) |
| 88–90 | Wrap-up | Read the one-sentence takeaway; assign one challenge | |

**If time is short:** §9 is self-contained. Show the occurrence-graph picture and the two classification reports (about 10 min), and leave the knowledge-graph challenge as homework or a mini-project.

## Expected results (our run; small differences are normal)
* Corpora: PubMed 236,879 sentences, 4.8 M tokens; Reuters 54,225 sentences, 1.3 M tokens.
* TF-IDF matrix: 20,000 abstracts x 32,797 terms, 0.32 % non-zero. PPMI: 2.38 % (PubMed) and 1.00 % (Reuters) non-zero.
* word2vec: about 2 min (PubMed) and 30 s (Reuters) on 2 cores.
* Semantic shift: 5,313 shared words, 1,119 frequent in both.
* WSD coarse accuracy, Reuters | PubMed: MFS 0.92 | 0.43 · domain MFS 0.92 | 0.57 · NLTK lesk 0.83 | 0.36 ·
  simplified Lesk 0.92 | 0.43 · GloVe 1.00 | 1.00 · domain word2vec 1.00 | 0.64.
* Entity linking (8 mentions): prior 4/8, dense GloVe 5/8, dense PubMed 5/8, dense+NIL 7/8, prior+dense+NIL 6/8
  (the NIL rule was chosen on these mentions: say so).
* Sense induction on *delivery*: 1,383 occurrences, 4 induced senses, Q = 0.57; on the 26 labelled ones purity 0.88, NMI 0.49, ARI 0.32.
* In PubMed, *she* is rare (20 vs. 134 for *he*), but *women* is three times more frequent than *men* (6,311 vs. 2,040).

## Likely questions
* **"Why are my neighbours different from yours?"** Multi-threaded word2vec is not deterministic. PPMI (§3) is.
* **"Is Procrustes cheating?"** It only rotates: distances inside each space are preserved. It assumes most words did *not* change meaning.
* **"Why does general GloVe beat PubMed word2vec at WSD?"** The sense signatures come from WordNet glosses written in general English; GloVe knows those words well, the PubMed model does not.
* **"Is a disease close to *female* a bias?"** Not necessarily: some conditions are more frequent in women. Association in a corpus is not the same as an unfair decision; the harm comes from how the vectors are used.

## Links to the rest of the module
* Slides *04.Semantics*: every section maps to the learning outcomes listed at the top of the notebook.
* Lab **AP04** covers the same concepts on Brown/SemCor with graded questions. This tutorial is the **demo**: use it in the lecture, AP04 in the lab.
