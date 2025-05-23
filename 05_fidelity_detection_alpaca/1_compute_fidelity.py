from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import pandas as pd

if __name__ == '__main__':
    model_names = [
        'ministral-8b-instruct-2410',
        'deepseek-r1-distill-llama-8b',
        'deepseek-r1-distill-qwen-7b',
        'qwen-2.5-7b_baseline'
    ]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    results_data = []
    for model_name in model_names:
            for threshold in thresholds:
                with open(f"../model_generations/alpaca/{model_name}.json", 'r') as f:
                    data = json.load(f)
                    original_outputs = [each['predict'] for each in data]

                with open(f"../model_generations/alpaca_sfi_{threshold}/{model_name}.json", 'r') as f:
                    data = json.load(f)
                    fogged_outputs = [each ['predict'] for each in data]

                original_embeddings = model.encode(original_outputs, convert_to_tensor=True)
                fogged_embeddings = model.encode(fogged_outputs, convert_to_tensor=True)
                cosine_scores = util.cos_sim(original_embeddings.to('cuda:0'), fogged_embeddings.to('cuda:0'))
                diagonal_scores = cosine_scores.diag().cpu().numpy()  # only matching pairs

                fidelity_score = float(np.mean(diagonal_scores))
                results_data.append([model_name, threshold, fidelity_score])
                print(f"model_name: {model_name}, threshold: {threshold}, fidelity score: {fidelity_score}")

    df_results = pd.DataFrame(results_data, columns=['model_name', 'threshold', 'fidelity_score'])
    df_results.to_csv("fidelity_scores.csv", index=False)