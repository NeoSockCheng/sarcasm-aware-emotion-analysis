import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime


def evaluate_task(model, dataloader, device, task_name):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, task=task_name)
            if task_name == "emotion":
                logits = outputs["emotion_logits"]
            elif task_name == "sarcasm":
                logits = outputs["sarcasm_logits"]
            else:
                raise ValueError("Invalid task name")

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "task": task_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labels": all_labels,
        "preds": all_preds
    }


def plot_confusion_matrix(y_true, y_pred, task_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{task_name.capitalize()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_multitask(model, emotion_loader, sarcasm_loader, device, save_dir=None, save=False):
    emotion_results = evaluate_task(model, emotion_loader, device, "emotion")
    sarcasm_results = evaluate_task(model, sarcasm_loader, device, "sarcasm")
    combined_f1 = (emotion_results["f1"] + sarcasm_results["f1"]) / 2

    result_data = {
        "emotion_metrics": {k: v for k, v in emotion_results.items() if k not in ["labels", "preds"]},
        "sarcasm_metrics": {k: v for k, v in sarcasm_results.items() if k not in ["labels", "preds"]},
        "combined_macro_f1": combined_f1
    }

    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_confusion_matrix(emotion_results["labels"], emotion_results["preds"],
                              "emotion", os.path.join(save_dir, f"emotion_cm_{timestamp}.png"))
        plot_confusion_matrix(sarcasm_results["labels"], sarcasm_results["preds"],
                              "sarcasm", os.path.join(save_dir, f"sarcasm_cm_{timestamp}.png"))

        with open(os.path.join(save_dir, f"eval_metrics_{timestamp}.json"), "w") as f:
            json.dump(result_data, f, indent=4)

    return result_data, combined_f1