import jsonargparse
from tqdm import tqdm 
import torch
import json 
from collections import defaultdict
import re
from vllm import LLM, SamplingParams  # 新增vLLM导入

from trained_calibration.rl.dataset.dataset import get_dataset
from trained_calibration.rl.reward_model import RewardModel
from trained_calibration.rl.dataset.postprocess import postprocess_answers, postprocess_extract

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def main(args):
    if args.split is None:
        split = "train"
    else:
        split = args.split

    dataset = get_dataset(args.dataset)
    if args.limit is not None:
        dataset_to_run = dataset[split].select(range(args.limit))
    else: 
        dataset_to_run = dataset[split]

    dataloader = torch.utils.data.DataLoader(
            dataset_to_run,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
    
    # 使用vLLM替换原始模型加载
    llm = LLM(
        model=args.model_name,
        tokenizer=args.model_name,
        tensor_parallel_size=len(args.model_device_map),  # 根据设备数量设置并行
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if "llama" in args.model_name:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_token = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        pad_token = tokenizer.unk_token_id
    tokenizer.padding_side = "left"

    # 配置生成参数
    generation_kwargs = SamplingParams(
        max_tokens=80,
        temperature=0.7,
        top_p=1.0,
        top_k=-1,  # 禁用top_k
        min_tokens=1,
        n=args.n_generations,  # 每个prompt生成多次
        skip_special_tokens=True
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    reward_models = [RewardModel(model_name, args.model_device_map[f'reward{i}'], quantization_config=bnb_config) 
                    for i, model_name in enumerate(args.reward_model_names)]

    with open(args.output_dir, "w") as f1:
        for batch in tqdm(dataloader):
            try:
                prompts = batch["generator_prompt"]
                
                # 使用vLLM批量生成（自动处理批处理）
                outputs = llm.generate(prompts, generation_kwargs)
                
                # 重组生成结果
                all_responses = []
                for output in outputs:
                    all_responses.extend([out.text for out in output.outputs])
                
                # 扩展prompts以匹配响应数量
                extended_prompts = [p for p in prompts for _ in range(args.n_generations)]
                
                # 后处理（假设postprocess_extract支持扩展后的批量处理）
                batch_responses_clean, batch_answers, batch_rationales = postprocess_extract(
                    extended_prompts, all_responses, None, tokenizer, args.dataset
                )

                responses_by_example = defaultdict(list)
                for idx in range(len(prompts)):
                    start_idx = idx * args.n_generations
                    end_idx = start_idx + args.n_generations
                    for i in range(start_idx, end_idx):
                        responses_by_example[idx].append({
                            "prompt": extended_prompts[i],
                            "response_clean": batch_responses_clean[i],
                            "response_orig": all_responses[i],
                            "answer": batch_answers[i]
                        })

                batch_questions = batch['evaluator_prompt']
                batch_correct_answers = [json.loads(x) for x in batch['correct_answer']]
                for query_idx, responses in responses_by_example.items():
                    question = batch_questions[query_idx]
                    correct_answers = batch_correct_answers[query_idx]
                    correct_answers = [correct_answers for _ in range(len(responses))]
                    question_batch = [question for _ in range(len(responses))]  
                    response_batch = [r["response_clean"] for r in responses]
                    answer_batch = [r["answer"] for r in responses]
                    # rationale_batch = [r["rationale"] for r in responses]

                    all_probs = []
                    for reward_model in reward_models:
                        if len(question_batch) > args.batch_size:
                            # chunk up
                            rewards, corrects, probs = [], [], []
                            for i in range(0, len(question_batch), args.batch_size):
                                rs, cs, ps, = reward_model(question_batch[i:i+args.batch_size], 
                                                                        response_batch[i:i+args.batch_size], 
                                                                        answer_batch[i:i+args.batch_size], 
                                                                        correct_answers[i:i+args.batch_size]) 
                                rewards.extend(rs)
                                corrects.extend(cs)
                                probs.extend(ps)

                        else:
                            rewards, corrects, probs, = reward_model(question_batch, response_batch, answer_batch, correct_answers) 
                        # vote over reward model probs
                        probs = [x.detach().cpu().item() for x in probs]
                        all_probs.append(probs)
                    all_probs = np.array(all_probs)
                    mean_probs = np.mean(all_probs, axis=0)

                    for response, mean_prob, all_prob, correct, correct_answer in zip(responses, mean_probs, all_probs.T, corrects, correct_answers):
                        response['all_probs'] = all_prob.tolist()
                        response['query_idx'] = query_idx 
                        response["mean_prob"] = mean_prob
                        response["correct"] = correct
                        response["correct_answers"] = json.dumps(correct_answer)
                        responses_by_example_final[query_idx].append(response)

                        f1.write(json.dumps(response) + "\n")
            except RuntimeError:
                print(f"Batch OOM, skipping")
                continue 

def extract_response(response):
    try:
        prompt, response = re.split("([Aa]nswer:)", response)
    except ValueError:
        return None
    # TODO (elias): remove incomplete sentences 
    return response.strip()

if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_argument("--cfg", action=jsonargparse.ActionConfigFile, help="path to config file")
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--reward_model_names", type=list, default=None, help="list of reward model names") 
    parser.add_argument("--model_device_map", type=dict, default="0", help="dict specifying which devices have which model")
    parser.add_argument("--dataset", type=str, default="trivia_qa")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_generations", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=6)

    args = parser.parse_args()

    main(args)