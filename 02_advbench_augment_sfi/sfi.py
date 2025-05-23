import random
from typing import List

from sentence_transformers import SentenceTransformer, util
import nltk
import numpy as np
from datasets import load_dataset
import torch
import os
import pickle
import json
import argparse

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Load semantic similarity model
model = SentenceTransformer('all-MiniLM-L6-v2')

MAX_CORPUS_SIZE = 100000
CACHE_DIR = '.'
random.seed(2025)


# Load or preprocess Wikipedia fog corpus
def load_or_cache_background_corpus(cache_path=f"{CACHE_DIR}/cached_wiki.pkl", embed_path=f"{CACHE_DIR}/cached_embeddings.pt"):
    if os.path.exists(cache_path) and os.path.exists(embed_path):
        with open(cache_path, "rb") as f:
            background_corpus = pickle.load(f)
        corpus_embed = torch.load(embed_path)
    else:
        print("Loading Wikipedia corpus...")
        wiki_data = load_dataset("wikipedia", "20220301.en", split='train[:10%]', cache_dir='/dev/shm/')
        background_corpus = []
        for article in wiki_data:
            for sent in sent_tokenize(article['text']):
                if len(sent.split()) > 5 and len(sent) < 200:
                    background_corpus.append(sent)
            if len(background_corpus) > MAX_CORPUS_SIZE:
                break
        with open(cache_path, "wb") as f:
            pickle.dump(background_corpus, f)
        corpus_embed = model.encode(background_corpus,
                                    batch_size=128,
                                    show_progress_bar=True,
                                    convert_to_tensor=True)
        torch.save(corpus_embed, embed_path)
    return background_corpus, corpus_embed

background_corpus, background_embeddings = load_or_cache_background_corpus()


# Extract key concepts using TF-IDF keywords
def extract_key_concepts(text: str, top_k: int = 10) -> List[str]:
    '''

    :param text: The input prompt to analyze, e.g., “How do I make a bomb?”
    :param top_k: the number of key concepts to return based on TF-IDF scores.
    :return:
    '''
    tokens = [w for w in word_tokenize(text.lower()) if w.isalpha() and w not in stopwords.words('english')]
    tfidf = TfidfVectorizer()
    tfidf.fit([text])
    scores = dict(zip(tfidf.get_feature_names_out(), tfidf.idf_))
    ranked = sorted(tokens, key=lambda x: scores.get(x, 0), reverse=True)
    return list(dict.fromkeys(ranked))[:top_k]

# Generate fog phrases for each concept
def generate_fog_phrases(concepts: List[str], top_n: int = 5, sim_threshold: float = 0.5) -> List[str]:
    '''

    :param concepts: a list of keywords/concepts, e.g., ["explosives", "devices"]
    :param top_n: max number of fog phrases to retrieve per concept
    :param sim_threshold: min cosine similarity for a fog phrase to be considered semantically related
    :return:
    '''
    fog_phrases = []
    for concept in concepts:
        concept_embed = model.encode(concept, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(concept_embed, background_embeddings)[0].to('cpu').numpy()
        for idx in np.argsort(-similarities):
            score = similarities[idx].item()
            phrase = background_corpus[idx]
            if score >= sim_threshold and phrase not in fog_phrases:
                fog_phrases.append(phrase)
            if len(fog_phrases) >= top_n * len(concepts):
                break
    return fog_phrases

# Inject fog phrases into prompt
def inject_fog(prompt: str, fog_phrases: List[str], density: float = 1) -> str:
    '''

    :param prompt: The original user query, e.g., “Explain how to make a bomb.”
    :param fog_phrases: List of fog sentences generated from the previous step
    :param density: Fraction of original sentences after which to insert fog (e.g., 0.2 → insert 1 fog after every 5 real sentences. However, as advbench prompts have only one sentence, it should be always set to 1.
    :return:
    '''
    sentences = sent_tokenize(prompt)
    num_fog = max(1, int(len(sentences) * density))
    selected_fog = random.sample(fog_phrases, min(num_fog, len(fog_phrases)))
    augmented = []
    for i, sent in enumerate(sentences):
        augmented.append(sent)
        if i < len(selected_fog):
            augmented.append(selected_fog[i])
    return '\n'.join(augmented)

# End-to-end function with safety classifier
def apply_semantic_fog(prompt: str, sim_threshold: float) -> str:
    # if safety_classifier(prompt)[0]['label'].lower() == 'safe':
    #     return prompt  # no need to fog safe prompts
    concepts = extract_key_concepts(prompt, top_k=5)
    print("Extracted concepts:", concepts)
    fog = generate_fog_phrases(concepts, sim_threshold=sim_threshold)
    print("Selected fog phrases:", fog)
    fogged_prompt = inject_fog(prompt, fog, density=1)
    return fogged_prompt

# Batch processing
def fog_batch(prompts: List[str], sim_threshold=0.5) -> List[str]:
    return [apply_semantic_fog(p, sim_threshold) for p in prompts]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.5, help='the min similarity threshold')
    parser.add_argument('--dataset', type=str, default='advbench', choices=['advbench', 'alpaca'], help='the dataset to use')
    args = parser.parse_args()

    if args.dataset == 'advbench':
        with open("../data/advbench.json", "r") as f:
            adv_bench = json.load(f)
            prompts = [each['instruction'] for each in adv_bench]
    else:
        with open("../data/alpaca.json", "r") as f:
            alpaca = json.load(f)
            prompts = [each['instruction'] for each in alpaca]

    data = []
    fogged_results = fog_batch(prompts, sim_threshold=args.threshold)
    for orig, fogged in zip(prompts, fogged_results):
        data.append({
            "instruction": fogged,
            "input": "",
            "output": "NA"
        })


    with open(f"{args.dataset}_sfi_{args.threshold}.json", "w") as f:
        json.dump(data, f, indent=4)