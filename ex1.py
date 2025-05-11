#!/usr/bin/env python3
import os
import argparse
import wandb
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,



    DataCollatorWithPadding
)
from transformers.trainer_callback import TrainerCallback
from sklearn.metrics import classification_report
import torch
import glob
import platform

# Argument parsing
parser = argparse.ArgumentParser(description="Fine-tune BERT on MRPC or run prediction.")
parser.add_argument("--max_train_samples", type=int, default=-1)
parser.add_argument("--max_eval_samples", type=int, default=-1)
parser.add_argument("--max_predict_samples", type=int, default=-1)
parser.add_argument("--num_train_epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_predict", action="store_true")
parser.add_argument("--do_predict_worst", action="store_true")

parser.add_argument("--model_path", type=str, default="best_model")
parser.add_argument("--use_mps", action="store_true", help="Use MPS backend on Apple Silicon")
args = parser.parse_args()

# Load dataset
dataset = load_dataset("glue", "mrpc")
if args.max_train_samples > 0:
    dataset["train"] = dataset["train"].select(range(args.max_train_samples))
if args.max_eval_samples > 0:
    dataset["validation"] = dataset["validation"].select(range(args.max_eval_samples))
if args.max_predict_samples > 0:
    dataset["test"] = dataset["test"].select(range(args.max_predict_samples))

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=tokenizer.model_max_length)

tokenized = dataset.map(preprocess_function, batched=True, remove_columns=["sentence1", "sentence2", "idx"])
tokenized.set_format(type="torch")

accuracy = evaluate.load("accuracy")

def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

class MetricsLogger(TrainerCallback):
    def __init__(self):
        self.logs = []
        self.train_losses = {}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs and "epoch" in logs:
            epoch = int(logs["epoch"])
            self.train_losses.setdefault(epoch, []).append(logs["loss"])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch_int = int(state.epoch)
        train_loss_list = self.train_losses.get(epoch_int)
        train_loss_avg = sum(train_loss_list) / len(train_loss_list) if train_loss_list else None
        if metrics:
            self.logs.append({
                "epoch": state.epoch,
                "eval_loss": metrics.get("eval_loss"),
                "eval_accuracy": metrics.get("eval_accuracy"),
                "train_loss": train_loss_avg
            })

best_model_ckpt = None
best_val_acc = -1
res_lines = []

if args.do_train:
    wandb.login()
    run = wandb.init(project="mrpc-paraphrase", config={"lr": args.lr, "batch_size": args.batch_size, "epochs": args.num_train_epochs})
    run.name = f"run_{args.num_train_epochs}_lr{args.lr}_bs{args.batch_size}"
    run.save()
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    use_mps = args.use_mps and torch.backends.mps.is_available() and platform.system() == "Darwin"
    no_cuda = not torch.cuda.is_available() and not use_mps

    output_dir = f"./results/run_{args.lr}_{args.batch_size}_{args.num_train_epochs}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=10,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        report_to=["wandb"],
        load_best_model_at_end=False,
        no_cuda=no_cuda
    )

    logger = MetricsLogger()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[logger]
    )

    trainer.train()
    # Save final model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    eval_results = trainer.evaluate()
    acc = eval_results.get("eval_accuracy", 0.0)

    # Evaluate on test set right after validation
    test_results = trainer.evaluate(eval_dataset=tokenized["test"])
    test_acc = test_results.get("eval_accuracy", 0.0)
    val_line = f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {acc:.4f}, test_acc: {test_acc:.4f}"
    res_lines.append(val_line)

    print("\n===== Training Summary Table =====")
    print("Epoch | Train Loss | Eval Loss | Accuracy")
    for log in logger.logs:
        epoch = int(log['epoch'])
        train_loss = f"{log['train_loss']:.6f}" if log['train_loss'] is not None else "   N/A  "
        eval_loss = f"{log['eval_loss']:.6f}" if log['eval_loss'] is not None else "   N/A  "
        accuracy = f"{log['eval_accuracy']:.6f}" if log['eval_accuracy'] is not None else "   N/A  "
        print(f"{epoch:5d} | {train_loss} | {eval_loss} | {accuracy}")

    if acc > best_val_acc:
        best_val_acc = acc
        best_model_ckpt = output_dir
        print(f" Using final model from: {best_model_ckpt}")
    else:
            print(f" No checkpoints found in {output_dir}")

    run.finish()
    with open("res.txt", "a") as f:
        for line in res_lines:
            f.write(line + "\n")

if args.do_predict:
    best_model_ckpt = None
    best_score = -1
    best_config = None
    print("\n===== Predict Summary Table =====")

    if os.path.exists("res.txt"):
        with open("res.txt", "r") as res_file:
            for line in res_file:
                try:
                    parts = line.strip().split(", ")
                    acc_str = [p for p in parts if p.startswith("eval_acc")][0]
                    acc = float(acc_str.split(": ")[1])
                    print(acc,best_score)
                    if acc > best_score:

                        best_score = acc
                        best_config = parts
                except Exception as e:
                    continue

    if best_config:
        config_dict = {kv.split(": ")[0]: kv.split(": ")[1] for kv in best_config}
        best_model_ckpt = f"./results/run_{config_dict['lr']}_{config_dict['batch_size']}_{config_dict['epoch_num']}"

    if best_model_ckpt and os.path.exists(best_model_ckpt):
        model = AutoModelForSequenceClassification.from_pretrained(best_model_ckpt)
        model.eval()
        original_test = dataset["test"]
        def preprocess_predict(examples):
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding=False, max_length=tokenizer.model_max_length)

        test_tokenized = original_test.map(preprocess_predict, batched=True, remove_columns=["sentence1", "sentence2", "idx"])
        test_tokenized.set_format(type="torch")
        trainer = Trainer(model=model, tokenizer=tokenizer)
        output = trainer.predict(test_tokenized)
        preds = output.predictions.argmax(axis=-1)

        with open("predictions.txt", "w", encoding="utf-8") as f:
            for s1, s2, pred in zip(original_test["sentence1"], original_test["sentence2"], preds):
                f.write(f"{s1.strip()}###{s2.strip()}###{int(pred)}\n")

if args.do_predict_worst:
    print("fvd")
    best_model_ckpt = None
    best_score = 100
    best_config = None

    if os.path.exists("res.txt"):
        with open("res.txt", "r") as res_file:
            for line in res_file:
                try:
                    parts = line.strip().split(", ")
                    acc_str = [p for p in parts if p.startswith("eval_acc")][0]
                    acc = float(acc_str.split(": ")[1])
                    print(acc_str,acc)
                    if acc < best_score:
                        best_score = acc
                        best_config = parts
                except Exception as e:
                    continue

    if best_config:
        config_dict = {kv.split(": ")[0]: kv.split(": ")[1] for kv in best_config}
        best_model_ckpt = f"./results/run_{config_dict['lr']}_{config_dict['batch_size']}_{config_dict['epoch_num']}"

    if best_model_ckpt and os.path.exists(best_model_ckpt):

        model = AutoModelForSequenceClassification.from_pretrained(best_model_ckpt)
        model.eval()

        original_test = dataset["test"]

        def preprocess_predict(examples):
            return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding=False, max_length=tokenizer.model_max_length)

        test_tokenized = original_test.map(preprocess_predict, batched=True, remove_columns=["sentence1", "sentence2", "idx"])
        test_tokenized.set_format(type="torch")
        trainer = Trainer(model=model, tokenizer=tokenizer)
        output = trainer.predict(test_tokenized)
        preds = output.predictions.argmax(axis=-1)

        with open("predictions_worst.txt", "w", encoding="utf-8") as f:
            for s1, s2, pred in zip(original_test["sentence1"], original_test["sentence2"], preds):
                f.write(f"{s1.strip()}###{s2.strip()}###{int(pred)}\n\n")
