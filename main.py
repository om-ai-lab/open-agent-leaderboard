import argparse
import json
import os
from datetime import datetime
from utils import *
from evaluation.datasets.gsm8k import Gsm8kEvaluator
from evaluation.datasets.aqua import AQuAEvaluator
from evaluation.datasets.hotpotqa import HotpotQaEvaluator
from evaluation.metrics.pass_rate import PassRateEvaluator
from evaluation.multi_evaluator import MultiEvaluator
from evaluation.datasets.math import MATHEvaluator
from evaluation.datasets.mme_realworld_lite import MMERealWorldLiteEvaluator


def parse_and_print_args():
    """
    Parse command line arguments and print them.
    Returns:
        Namespace: Parsed arguments.
    """
    args = parse_arguments()
    print("*****************************")
    print(args)
    print("*****************************")
    return args

def load_predictions(args):
    """
    Load model predictions from a JSON file.
    Args:
        args (Namespace): Arguments containing the output path.
    Returns:
        list: List of model predictions.
    """
    with open(args.output_path, "r", encoding="utf-8") as f:
        return json.load(f)["model_result"]

def handle_model_output(args, question, model_output, decoder):
    """
    Handle the model output based on the specified method.
    Args:
        args (Namespace): Arguments containing various settings and configurations.
        question (str): The input question.
        model_output (str): The raw model output.
        decoder (object): Decoder object used for handling model output.
    Returns:
        str: Cleaned model output.
    """
    if not isinstance(model_output, str):
        model_output = str(model_output)
    if args.method == "zero_shot_cot":
        model_output_postprocess = f"Question: {question}\nModel answer: {model_output} {args.direct_answer_trigger_for_zeroshot_cot}"
        model_output, _, _ = decoder.decode(args, model_output_postprocess, args.max_length_direct, 0, 2)
    if args.dataset == "math500":
        model_output = model_output.split("Final Answer: ")[-1] if "Final Answer: " in model_output else model_output
    return answer_cleansing(args, model_output)

def process_predictions(pred_data, args, decoder, gt_data):
    """
    Processes prediction data and compares it with ground truth data.
    Args:
        pred_data (list): A list of dictionaries containing prediction data.
        args (Namespace): Arguments containing various settings and configurations.
        decoder (object): Decoder object used for handling model output.
        gt_data (dict): Dictionary containing ground truth data.
    Returns:
        tuple: A tuple containing:
            - predictions (list): List of processed predictions.
            - references (list): List of ground truth references.
            - input_tokens (list): List of input token counts.
            - output_tokens (list): List of output token counts.
            - total (int): Total number of processed entries.
    Note:
        If the ground truth answer is not found using the entry_id, it attempts to convert entry_id to an integer and search again.
        If args.limit_dataset_size is set to a non-zero value, the processing will stop once the limit is reached.
    """
    predictions, references, input_tokens, output_tokens, total_tokens = [], [], [], [], []
    total = 0

    for i, pre in enumerate(pred_data):
        model_output = pre.get("last_output", "")
        question = pre.get("question", "")
        entry_id = pre["id"]
        input_tokens.append(pre.get("prompt_tokens", 0))
        output_tokens.append(pre.get("completion_tokens", 0))
        total_tokens.append(pre.get("total_tokens", 0))
        
        if model_output:
            cleaned_answer = handle_model_output(args, question, model_output, decoder)
        else:
            cleaned_answer = answer_cleansing(args, model_output)
        
        pred = floatify_ans(cleaned_answer)
        predictions.append(pred)
        
        gt_answer = gt_data.get(entry_id)
        if gt_answer is None:
            try:
                entry_id_int = int(entry_id)
                gt_answer = gt_data.get(entry_id_int)
            except ValueError:
                pass
        
        references.append(gt_answer)

        print(f"======== id : {entry_id}, pred : {pred}, gt : {gt_answer}")
        total += 1

        if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
            break

    return predictions, references, input_tokens, output_tokens, total

def calculate_token_statistics(args, input_tokens, output_tokens):
    """
    Calculate and print token statistics and cost.
    Args:
        args (Namespace): Arguments containing various settings and configurations.
        input_tokens (list): List of input token counts.
        output_tokens (list): List of output token counts.
    """
    exchange_rate = 7.3249
    if args.model == "Doubao-lite-32k":
        input_token_cost = 0.0003 / exchange_rate # dollars / 1K tokens 
        output_token_cost = 0.0006 / exchange_rate # dollars / 1K tokens 
    elif args.model == "gpt-3.5-turbo":
        input_token_cost = 0.0005  # dollars / 1K tokens 
        output_token_cost = 0.0015  # dollars / 1K tokens 
    total_input_tokens = sum(input_tokens)
    average_input_tokens = total_input_tokens / len(input_tokens) if input_tokens else 0
    total_output_tokens = sum(output_tokens)
    average_output_tokens = total_output_tokens / len(output_tokens) if output_tokens else 0
    total_tokens_sum = total_input_tokens + total_output_tokens

    input_cost = total_input_tokens / 1000 * input_token_cost
    output_cost = total_output_tokens / 1000 * output_token_cost
    total_cost = input_cost + output_cost  # dollars

    print("Total input tokens:", total_input_tokens)
    print("Average input tokens:", average_input_tokens)
    print("Total output tokens:", total_output_tokens)
    print("Average output tokens:", average_output_tokens)
    print("Total tokens:", total_tokens_sum)
    print("Total cost for tokens:", total_cost)


def main():
    """
    Main function to run the evaluation process.
    """
    args = parse_and_print_args()

    fix_seed(args.random_seed)
    print("OPENAI_API_KEY:\n", os.getenv("OPENAI_API_KEY"))

    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder(args)
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    # Initialize the evaluator
    dataset_evaluator = {
        "aqua": AQuAEvaluator(),
        "gsm8k": Gsm8kEvaluator(),
        "hotpotqa": HotpotQaEvaluator(),
        "math500": MATHEvaluator(),
        "mme_realworld_lite": MMERealWorldLiteEvaluator(),
    }.get(args.dataset)

    metric_evaluator = PassRateEvaluator()
    
    multi_evaluator = MultiEvaluator([dataset_evaluator, metric_evaluator])
    
    json_results = {
        "dataset": args.dataset,
        "model_id": args.model,
        "alg": args.agent,
        "model_result": [],
    }
    

    if args.output_path:
        pred_data = load_predictions(args)
        gt_data = load_ground_truth(args.dataset_path, args.dataset)
        predictions, references, input_tokens, output_tokens, total = process_predictions(pred_data, args, decoder, gt_data)

    else:
        total = 0
        demo = create_demo_text(args, cot_flag=(args.method == "few_shot_cot")) if args.method in ["few_shot", "few_shot_cot"] else ""
        
        for i, data in enumerate(dataloader):
            print(f"*************************\n{i + 1}st data")
            x, y, idx = data
            x = "Q: " + x[0] + "\nA:"
            y = y[0].strip()

            if args.method == "zero_shot":
                x = x + " " + args.direct_answer_trigger_for_zeroshot
            elif args.method == "zero_shot_cot":
                x = x + " " + args.cot_trigger
            elif args.method in ["few_shot", "few_shot_cot"]:
                x = demo + x

            max_length = args.max_length if "cot" not in args.method else args.max_length_direct
            
            z, _, raw_full_response = decoder.decode(args, x, max_length, i, 1)

            # Answer extraction for zero-shot-cot ...
            if args.method == "zero_shot_cot":
                z2 = x + z + " " + args.direct_answer_trigger_for_zeroshot_cot
                pred, _, _ = decoder.decode(args, z2, args.max_length_direct, i, 2)
                print(z2 + pred)
            else:
                pred = z
                print(x + pred)

            # Clensing of predicted answer ...
            pred = answer_cleansing(args, pred)
            predictions.append(pred)
            references.append(y)

            # Token statistics
            input_token = raw_full_response.json()["usage"]["prompt_tokens"]
            output_token = raw_full_response.json()["usage"]["completion_tokens"]
            input_tokens.append(input_token)
            output_tokens.append(output_token)

            print("pred : {}".format(pred))
            print("GT : " + y)
            print("*************************")

            result = {
                "id": idx[0].item(),
                "question": x,
                "last_output": z,
                "output_postprocess": pred,
                "ground_truth": y,
                "prompt_tokens": input_token,
                "completion_tokens": output_token,
            }
            json_results["model_result"].append(result)

            # Save current results to JSON file
            save_dir = os.path.join(args.output_dir, args.agent, args.dataset)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, f"{args.method}_{args.limit_dataset_size}_{args.model}_{timestamp}.json"), "w") as ans_file:
                json.dump(json_results, ans_file, indent=4)
                ans_file.flush()
                ans_file.close()
                
            total += 1
            if (args.limit_dataset_size != 0) and ((i + 1) >= args.limit_dataset_size):
                break

    # Evaluate predictions using the evaluator
    evaluation_results = multi_evaluator.evaluate(predictions, references)
    print("Evaluation Result:", evaluation_results)

    if args.dataset == "hotpotqa":
        print("Reward : {}".format(evaluation_results.get("reward", 0)))
        print("F1 : {}".format(evaluation_results.get("f1", 0)))
        print("EM : {}".format(evaluation_results.get("em", 0)))
    else:
        # Calculate accuracy ...
        print("Accuracy : {}".format(evaluation_results.get("accuracy", 0)))
    # Calculate pass rate ...
    print("Pass Rate : {}\n".format(evaluation_results.get("pass_rate", 0)))
    print(f" ======  Testing {args.agent} on {args.dataset} dataset")

    print("Total predictions:", total)
    calculate_token_statistics(args, input_tokens, output_tokens)

def parse_arguments():
    """
    Parse command line arguments.
    Returns:
        Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="OmAgent Evaluation")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="aqua",
        choices=[
            "aqua",
            "gsm8k",
            "hotpotqa",
            "math500",
            "mme_realworld_lite",
        ],
        help="dataset used for experiment",
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=1,
        choices=[1],
        help="minibatch size (default 1)",
    )
    parser.add_argument(
        "--max_num_worker",
        type=int,
        default=4,
        help="maximum number of workers for dataloader",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        choices=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
            "Doubao-lite-32k",
        ],
        help="model used for decoding.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="zero_shot_cot",
        choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot"],
        help="method",
    )
    parser.add_argument(
        "--cot_trigger_no",
        type=int,
        default=1,
        help="A trigger sentence that elicits a model to execute chain of thought",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="maximum length of output tokens by model for reasoning extraction",
    )
    parser.add_argument(
        "--max_length_direct",
        type=int,
        default=32,
        help="maximum length of output tokens by model for answer extraction",
    )
    parser.add_argument(
        "--limit_dataset_size",
        type=int,
        default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs/", help="output directory"
    )
    parser.add_argument("--output_path", type=str, default=None, help="output path")
    parser.add_argument(
        "--agent",
        type=str,
        default="cot",
        choices=["cot", "pot", "sc_cot", "react", "dnc", "tot"],
        help="agent used for experiment",
    )
    parser.add_argument("--system_prompt", type=str, default="", help="system prompt")
    parser.add_argument("--openai_api_key", type=str, default="sk-xxx", help="openai api key")
    parser.add_argument("--openai_url", type=str, default="https://api.openai.com/v1", help="openai url")

    args = parser.parse_args()

    configure_dataset(args)
    configure_cot_trigger(args)

    return args


if __name__ == "__main__":
    main()
