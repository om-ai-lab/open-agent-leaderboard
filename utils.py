from statistics import mean
from torch.utils.data import Dataset
import os
import multiprocessing
import json
import numpy as np
import random
import torch
import re
import random
import time
import datetime
import requests


def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def print_now(return_flag=0):
    t_delta = datetime.timedelta(hours=8)
    CST = datetime.timezone(t_delta, "CST")
    now = datetime.datetime.now(CST)
    now = now.strftime("%Y/%m/%d %H:%M:%S")
    if return_flag == 0:
        print(now)
    elif return_flag == 1:
        return now
    else:
        pass


# Sentence Generator (Decoder) for GPT 
def decoder_for_gpt(args, input, max_length, i, k):
    time.sleep(1)

    engine_map = {
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4o": "gpt-4o",
        "gpt-3.5-turbo": "gpt-3.5-turbo"
    }
    engine = engine_map.get(args.model)
    if engine is None:
        raise ValueError("Model is not properly defined ")
    
    payload = {
        "model": engine,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input,
                    }
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": max_length,
    }
    
    if args.system_prompt != "":
        payload["messages"].insert(0, {"role": "system", "content": args.system_prompt})
    
    headers = {"Authorization": f"Bearer {args.openai_api_key}", "Content-Type": "application/json"}
    try:
        response = requests.post(args.openai_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an error for bad responses
        output = response.json()["choices"][0]["message"]["content"]

        return output, payload, response
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the API: {e}")
        return None, payload, None


class Decoder:
    def __init__(self, args):
        print_now()

    def decode(self, args, input, max_length, i, k):
        response, payload, full_response = decoder_for_gpt(
            args, input, max_length, i, k
        )
        return response, payload, full_response


def data_reader(args):
    ids = []
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + " " + choice)
                answers.append(json_res["correct"])
                ids.append(json_res["id"])

    elif args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1])
                ids.append(json_res["id"])

    elif args.dataset == "hotpotqa":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                questions.append(line["question"].strip())
                answers.append(line["answer"].strip())
                ids.append(line["_id"]) 
    else:
        raise ValueError("dataset is not properly defined ")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))

    return questions, answers, ids


# Create dataset object before dataloader 
class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers, self.ids = data_reader(args)
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        id = self.ids[index]
        input = self.questions[index]
        output = self.answers[index]
        return input, output, id


def setup_data_loader(args):

    # fix randomness of dataloader to ensure reproducibility
    # https://pytorch.org/docs/stable/notes/randomness.html
    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))

    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataloader_num_workers = multiprocessing.cpu_count()
    dataloader_num_workers = min(dataloader_num_workers, args.max_num_worker)
    print("dataloader_num_workers: " + str(dataloader_num_workers))

    dataset = MyDataset(args)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.minibatch_size,
        drop_last=False,
        num_workers=dataloader_num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        pin_memory=True,
    )

    return dataloader


# ver 0.2
def answer_cleansing(args, pred):

    print("pred_before : " + str(pred))

    if args.method in ("few_shot", "few_shot_cot"):
        if isinstance(pred, (int, float)):
            pred = str(pred)
            answer_flag = False
        else:
            preds = pred.split(args.direct_answer_trigger_for_fewshot)
            answer_flag = True if len(preds) > 1 else False
            pred = preds[-1]

    if args.dataset in ("aqua", ):
        pred = re.findall(r"A|B|C|D|E", pred)
    elif args.dataset in ("gsm8k", ):
        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]
    elif args.dataset == "hotpotqa":
        pass
    else:
        raise ValueError("dataset is not properly defined ")

    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        if args.method in ("few_shot", "few_shot_cot"):
            if answer_flag:
                # choose the first element in list 
                pred = pred[0]
            else:
                # choose the last element in list 
                pred = pred[-1]
        elif args.method in ("zero_shot", "zero_shot_cot"):
            # choose the first element in list 
            pred = pred[0]
        else:
            raise ValueError("method is not properly defined ")

    # (For arithmetic tasks) if a word ends with period, it will be omitted 
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]

    print("pred_after : " + pred)

    return pred


def floatify_ans(ans):
    if ans is None or ans == "None" or ans == "":
        return None
    # Handle dictionary type
    elif isinstance(ans, dict):
        ans = list(ans.values())[0]  # Get the first value from the dictionary
    # Handle boolean type
    elif isinstance(ans, bool):
        ans = ans
    # Handle list or tuple
    elif isinstance(ans, (list, tuple)):
        if not ans:  # If the list or tuple is empty
            return None
        else:
            try:
                ans = int(float(ans[0]))
            except (ValueError, TypeError):
                ans = int(str(ans[0]))
    else:
        try:
            if isinstance(ans, str):
                # If it's an option (e.g., 'A', 'B', 'C', 'D', 'E'), return it directly
                if ans in ["A", "B", "C", "D", "E"]:
                    return ans
            return int(float(ans))
        except (ValueError, TypeError):
            ans = int(str(ans))
    return ans


def create_demo_text(args, cot_flag):
    x, z, y = [], [], []

    # example sentences 
    if args.dataset in ("gsm8k", ):

        x.append(
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
        )
        z.append(
            "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6."
        )
        y.append("6")

        x.append(
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
        )
        z.append("There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.")
        y.append("5")

        x.append(
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
        )
        z.append(
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39."
        )
        y.append("39")

        x.append(
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"
        )
        z.append(
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8."
        )
        y.append("8")

        x.append(
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"
        )
        z.append(
            "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9."
        )
        y.append("9")

        x.append(
            "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"
        )
        z.append(
            "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29."
        )
        y.append("29")

        x.append(
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"
        )
        z.append(
            "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls."
        )
        y.append("33")

        x.append(
            "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
        )
        z.append(
            "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8."
        )
        y.append("8")

    else:
        raise ValueError("dataset is not properly defined ")

    # randomize order of the examples 
    index_list = list(range(len(x)))

    # Concatenate demonstration examples 
    demo_text = ""
    for i in index_list:
        if cot_flag:
            demo_text += (
                "Q: "
                + x[i]
                + "\nA: "
                + z[i]
                + " "
                + args.direct_answer_trigger_for_fewshot
                + " "
                + y[i]
                + ".\n\n"
            )
        else:
            demo_text += (
                "Q: "
                + x[i]
                + "\nA: "
                + args.direct_answer_trigger_for_fewshot
                + " "
                + y[i]
                + ".\n\n"
            )

    return demo_text


def configure_dataset(args):
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "hotpotqa":
        args.dataset_path = "./dataset/hotpotqa/hotpot_dev_distractor_v1.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ")

    return args


def configure_cot_trigger(args):
    # "Therefore, the answer " -> "The answer "
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger

    args.direct_answer_trigger_for_fewshot = "The answer is"

    if args.cot_trigger_no == 1:
        args.cot_trigger = "Let's think step by step."
    elif args.cot_trigger_no == 2:
        args.cot_trigger = "We should think about this step by step."
    elif args.cot_trigger_no == 3:
        args.cot_trigger = "First,"
    elif args.cot_trigger_no == 4:
        args.cot_trigger = "Before we dive into the answer,"
    elif args.cot_trigger_no == 5:
        args.cot_trigger = "Proof followed by the answer."
    elif args.cot_trigger_no == 6:
        args.cot_trigger = "Let's think step by step in a realistic way."
    elif args.cot_trigger_no == 7:
        args.cot_trigger = "Let's think step by step using common sense and knowledge."
    elif args.cot_trigger_no == 8:
        args.cot_trigger = "Let's think like a detective step by step."
    elif args.cot_trigger_no == 9:
        args.cot_trigger = "Let's think about this logically."
    elif args.cot_trigger_no == 10:
        args.cot_trigger = "Let's think step by step. First,"
    elif args.cot_trigger_no == 11:
        args.cot_trigger = "Let's think"
    elif args.cot_trigger_no == 12:
        args.cot_trigger = "Let's solve this problem by splitting it into steps."
    elif args.cot_trigger_no == 13:
        args.cot_trigger = "The answer is after the proof."
    elif args.cot_trigger_no == 14:
        args.cot_trigger = "Let's be realistic and think step by step."
    else:
        raise ValueError("cot_trigger_no is not properly defined ")

    return args


def load_ground_truth(dataset_path: str) -> dict:
    """Load ground truth data from the dataset path."""
    gt_data = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            gt_data[entry["id"]] = entry
    return gt_data