### Run batch inference on the original dataset and fog-injected datasets using different LLMs
- Ministral-8B-Instruct-2410
- Deepseek-R1-distill-Llama-8B
- Deepseek-R1-distill-Qwen-7B
- Qwen-2.5-7B

We use [vLLM](https://github.com/vllm-project/vllm) implemented [in LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main).
Below are some examples command.

```angular2html
CUDA_VISIBLE_DEVICES=0 python scripts/vllm_infer.py \
  --model_name_or_path=mistralai/Ministral-8B-Instruct-2410 \
  --dataset=advbench \
  --template=mistral \
  --save_name=../semantic-fog-injection/model_generations/advbench/ministral-8b-instruct-2410.jsonl \
  --top_p=0.9 \
  --temperature=0


CUDA_VISIBLE_DEVICES=0 python scripts/vllm_infer.py \
  --model_name_or_path=mistralai/Ministral-8B-Instruct-2410 \
  --dataset=advbench_sfi_0.1 \
  --template=mistral \
  --save_name=../semantic-fog-injection/model_generations/advbench_sfi_0.1/ministral-8b-instruct-2410.jsonl \
  --top_p=0.9 \
  --temperature=0
```

The commands should be run under the LLaMA-Factory directory. Run it for all models and datasets, including the original ones and the fog-injected ones.
All the model generations are saved under *model_generations/*.