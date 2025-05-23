from datasets import load_dataset
import json


if __name__ == '__main__':
    dataset = load_dataset("walledai/AdvBench")

    # Convert to the required JSON format, in order to be able to be processed by Llama-Factory
    data = []
    for prompt, output in zip(dataset['train']['prompt'], dataset['train']['target']):
        data.append({
            "instruction": prompt,
            "input": "",
            "output": output
        })

    # Save to a JSON file
    with open("../data/advbench.json", "w") as f:
        json.dump(data, f, indent=4)

    print("JSON file created successfully!")
