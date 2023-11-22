import sys
import json

from transformers import AutoTokenizer


def get_token_len(text: str, tokenizer: AutoTokenizer):
    return len(tokenizer.encode(text))


def read_json_file_total_token(path: str) -> int:
    total_token = 0
    with open(path, "r") as f:
        data_obj = json.load(f)
        for obj in data_obj:
            text = ""
            for k in obj:
                if isinstance(obj[k], list):
                    for ll in obj[k]:
                        text += str(ll)
                else:
                    text += obj[k]
            token_len = get_token_len(text, tokenizer)
            total_token += token_len
    return total_token


def read_json_file(path: str, output: str):
    count = {}
    with open(path, "r") as f:
        data_obj = json.load(f)
        for obj in data_obj:
            text = ""
            for k in obj:
                if isinstance(obj[k], list):
                    for ll in obj[k]:
                        text += str(ll)
                else:
                    text += obj[k]
            token_len = get_token_len(text, tokenizer)
            if token_len not in count:
                count[token_len] = 0
            count[token_len] += 1

    with open(output, "w") as f:
        f.write(json.dumps(count))


def read_jsonl_file(path: str, output: str):
    count = {}
    with open(path, "r") as f:
        data_obj = [json.loads(line) for line in f]
        for obj in data_obj:
            text = ""
            for k in obj:
                if isinstance(obj[k], list):
                    text += ','.join(obj[k])
                else:
                    text += obj[k]
            token_len = get_token_len(text, tokenizer)
            if token_len not in count:
                count[token_len] = 0
            count[token_len] += 1

    with open(output, "w") as f:
        f.write(json.dumps(count))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: tokens_count.py [tokenizer_path]")
        exit(-1)

    tokenizer_path = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # read_jsonl_file(
    #     "/home/yezhengmao/train_data/llama-python-codes-30k-cleaned.jsonl", "llama_python_codes_token_count.json")

    # read_json_file(
    #     "/home/yezhengmao/train_data/alpaca_gpt4_data.json",
    #     "alpaca_token_count.json")

    # read_jsonl_file("/home/yezhengmao/train_data/train.jsonl",
    #                 "train_count.json")

    # read_json_file(
    #     "/home/yezhengmao/train_data/llava_instruct_150k.json", "llava_token_count.json")

    # read_json_file("/home/yezhengmao/train_data/sql_create_context_v4.json",
    #              "sql_create_token_count.json")

    print(read_json_file_total_token(
        "/home/yezhengmao/train_data/data_set_1.json"))
    print(read_json_file_total_token(
        "/home/yezhengmao/train_data/data_set_2.json"))
    print(read_json_file_total_token(
        "/home/yezhengmao/train_data/data_set_3.json"))
    print(read_json_file_total_token(
        "/home/yezhengmao/train_data/data_set_4.json"))
