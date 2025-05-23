from datasets import load_dataset
import json


if __name__ == '__main__':
    dataset = load_dataset("tatsu-lab/alpaca")

    # sample 200 demonstrations from alpaca that don't have "input"
    # Convert to the required JSON format, in order to be able to be processed by Llama-Factory
    data = []
    for prompt, input, output in zip(dataset['train']['instruction'], dataset['train']['input'], dataset['train']['output']):
        if input != '':
            continue
        data.append({
            "instruction": prompt,
            "input": "",
            "output": output
        })
        if len(data) >= 200:
            break

    # Save to a JSON file
    with open("../data/alpaca.json", "w") as f:
        json.dump(data, f, indent=4)

    print("JSON file created successfully!")
