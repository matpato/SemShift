# Instructor guide: SemShift practical lecture (≈ 90 min)

## Before class (10 min, the day before or that morning)
1. Open the notebook in Colab and run **§0 Setup** plus the GloVe cell once, to check that Colab, NLTK and gensim still work.
2. Keep the GitHub page of the notebook open in a second tab. It is saved **with outputs**, so it is the fallback if Colab is slow or the Wi-Fi fails.
3. At the start of class, ask the students to open the Colab link and run **§0 + the two training cells of §4** straight away (≈ 2 min). By the time you reach §4, their models are ready.

## Lesson plan

| Time | Section | What to do live | The "aha" to land |
|---|---|---|---|
| 0–5 | Hook | Show the README table (*bank, light, mine, house, tender*) and ask: *"Same word, same algorithm. Why different neighbours?"* | Meaning comes from the data |
| 5–15 | §1 WordNet | Run the senses of *bank*, the two hypernym paths, `lowest_common_hypernyms` → `entity.n.01`, *banco* in Portuguese | Homonymy vs. polysemy; sense 1 is the **river** bank (remember for §6) |
| 15–22 | §2 KWIC | Run the concordance and the context counts. Ask the 🔮 question before running | The distributional hypothesis with zero maths |
| 22–35 | §3 Sparse | TF-IDF heatmap: *"nobody told it the authors"*. Then the by-hand cosine and counts vs. PPMI | Cosine; why PPMI beats counts (*light* = sweet crude!) |
| 35–50 | §4 Dense | Explain the SGNS table while it trains. Show the three-worlds table and the PCA picture | One vector per word, decided by the corpus |
| 50–58 | §5 Shift ⭐ | Run the Procrustes cell. Ask the students to explain *tender, chase, fed* | Unsupervised discovery of polysemy; stable words = numbers and modals |
| 58–70 | §6 WSD | §6.1 predominant senses, then the 5-method table: let them **bet** on the winner first | **MFS = 0 % on Reuters**: frequency baselines are domain dependent |
| 70–80 | §7 EL | *Jordan* table: prior vs. dense vs. dense+NIL; then the live Wikidata cell | WSD vs. EL; NIL; the Wikipedia prior is wrong for the Bible |
| 80–88 | §8 Limits | *bank − money + river*, *hot/cold*, the bias chart (*maid, governess* vs. *president*; *clerk* flips) | A vector is a mirror of its corpus |
| 88–90 | Wrap-up | Read the one-sentence takeaway; assign one ✏️ challenge | |

§9 (Portuguese) is a good **5-minute bonus** if time allows, or homework. *rede* (hammock → internet) always gets a reaction.

## Expected results (our run; small differences are normal)
* TF-IDF: Austen ↔ Austen 0.70–0.72, Shakespeare ↔ Shakespeare 0.61–0.65, Bible ↔ Milton 0.50.
* PPMI matrices: 1.0 % (Reuters) and 1.4 % (Gutenberg) non-zero.
* word2vec: ≈ 35 s (Reuters) and ≈ 55 s (Gutenberg) on 2 cores.
* WSD accuracy (fine / coarse), Reuters | Gutenberg:
  MFS 0.00/0.00 | 0.67/0.83 · domain MFS 1.00/1.00 | 0.67/0.83 · NLTK lesk 0.17/0.67 | 0.25/0.50 ·
  simplified Lesk 0.17/0.42 | 0.42/0.75 · embeddings 0.17/1.00 | 0.33/1.00.
* Entity linking: prior 2/9, dense 6/9, dense+NIL 9/9 (the NIL rule was chosen on these 9 mentions: say so).
* `she` is ≈ 30× rarer than `he` in Reuters (163 vs. 5,208).

## Likely questions
* **"Why are my neighbours different from yours?"** Multi-threaded word2vec is not deterministic. PPMI (§3) is.
* **"Why does `mine` fail in §6.1?"** In the Bible it is a pronoun (*thine and mine*), and WordNet has no pronouns: a part-of-speech problem, not a sense problem.
* **"Is Procrustes cheating?"** It only rotates: distances inside each space are preserved. It assumes most words did *not* change meaning.
* **"Why does GloVe put `bank + river` near *central*?"** The financial sense dominates the vector, so you have to subtract *money*. That is why we need contextual embeddings (BERT, later in the course).

## Links to the rest of the module
* Slides *PLN_04_Semantics*: every section maps to the learning outcomes listed at the top of the notebook.
* Lab **AP04** covers the same concepts on Brown/SemCor with graded questions. This tutorial is the **demo**: use it in the lecture, AP04 in the lab.
