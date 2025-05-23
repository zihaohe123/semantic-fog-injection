For the responses to fog-injected prompts, we compute their fidelity scores, by comparing to the responses to the original prompts. The cosine similarities between their Sentence-BERT embeddings are used as the fidelity scores.
```angular2html
python 1_compute_fidelity.py
```
It computes the fidelity scores of responses from all LLMs to all fog-injected datasets. 