import json
from datasets import load_dataset
from sal.utils.math import memoized_canonical_form
from sal.utils.qwen_math_parser import extract_answer 

def compute_acc(dataset_path, pred_type):
    dataset = load_dataset("json", data_files=dataset_path)
    n_questions = dataset['train'].num_rows
    correct_counter = 0
    for x in dataset["train"]:
        pred = extract_answer(x["pred_weighted@4"], "math")
        canonical_pred = memoized_canonical_form(pred)
        canonical_answer = memoized_canonical_form(x["answer"])
        if canonical_pred == canonical_answer:
            correct_counter+=1
    return correct_counter, n_questions

if __name__ == "__main__":
    # dataset_path = "/home/huidong/search-and-learn/data/meta-llama/Llama-3.2-1B-Instruct/best_of_n_completions.jsonl"
    inference_model = "meta-llama/Llama-3.2-1B-Instruct"
    search_method = "best_of_n"
    dataset_path = "data/" + inference_model + f"/{search_method}_completions.jsonl"
    pred_type = "pred_weighted@4"
    n_correct, n_questions = compute_acc(dataset_path, pred_type)
    print(
        f"Inference Model: {inference_model}; Search Method: {search_method}; "
        f"Prediction Type: {pred_type}; Acc: {(n_correct/n_questions*100):.1f} - ({n_correct}/{n_questions})"
    )