import json
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score


# ---------------------------
# Generate Synthetic QA Data
# ---------------------------
def generate_synthetic_qa(n=10) -> List[Dict]:
    people = ["Messi",", "Ronaldo", "Dembele", "Mbappe"]
    cities = ["United States", "UAE",  "Paris", "Spain"]
    qa_data = []

    for _ in range(n):
        person = random.choice(people)
        city = random.choice(cities)
        context = f"{person} was born in {city}."
        question = f"Where was {person} born?"
        answer = city
        qa_data.append({
            "context": context,
            "question": question,
            "answer": answer
        })
    return qa_data


# ---------------------------
# Save to JSONL
# ---------------------------
def save_to_jsonl(data: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# ---------------------------
# Simulate QA Predictions
# ---------------------------
def simulate_predictions(data: List[Dict]) -> List[str]:
    # Simulate predictions (perfect or noisy)
    predictions = []
    for item in data:
        # 80% chance to get it right
        if random.random() < 0.8:
            predictions.append(item["answer"])
        else:
            predictions.append("Unknown")  
    return predictions


# ---------------------------
# Evaluation Metrics
# ---------------------------
def exact_match(pred: str, gold: str) -> int:
    return int(pred.strip().lower() == gold.strip().lower())

def f1_score(pred: str, gold: str) -> float:
    pred_tokens = pred.lower().split()
    gold_tokens = gold.lower().split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def evaluate(predictions: List[str], golds: List[str]):
    results = defaultdict(list)
    for pred, gold in zip(predictions, golds):
        results["exact_match"].append(exact_match(pred, gold))
        results["f1"].append(f1_score(pred, gold))
    
    # Average
    avg_em = sum(results["exact_match"]) / len(results["exact_match"])
    avg_f1 = sum(results["f1"]) / len(results["f1"])

    # BERTScore
    P, R, F1 = bert_score(predictions, golds, lang="en", verbose=False)
    avg_bert_f1 = F1.mean().item()

    return {
        "Exact Match": round(avg_em * 100, 2),
        "F1 Score": round(avg_f1 * 100, 2),
        "BERTScore F1": round(avg_bert_f1 * 100, 2)
    }


# ---------------------------
# Run Everything
# ---------------------------
if __name__ == "__main__":
    import nltk
    nltk.download("punkt")

    print("Generating synthetic QA data...")
    qa_data = generate_synthetic_qa(n=20)

    data_path = "synthetic_qa_dataset.jsonl"
    save_to_jsonl(qa_data, data_path)
    print(f"Saved synthetic QA data to {data_path}")

    print("Simulating predictions...")
    gold_answers = [item["answer"] for item in qa_data]
    predicted_answers = simulate_predictions(qa_data)

    print("Evaluating predictions...")
    scores = evaluate(predicted_answers, gold_answers)
    for k, v in scores.items():
        print(f"{k}: {v}%")
