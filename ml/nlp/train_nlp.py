"""
train_nlp.py — Fine-tune RoBERTa on GoEmotions for multi-label emotion classification.
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
import random
import numpy as np
import torch
import torch.nn as nn

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    default_data_collator,
)
from sklearn.metrics import f1_score

set_seed(SEED)

MODEL_NAME    = "roberta-base"
OUTPUT_DIR    = "models/roberta_scratch_goemotions"   # NOT models/roberta_finetuned: that folder is the app's model
MAX_LENGTH    = 128
NUM_LABELS    = 28
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS    = 5
POS_WEIGHT_CAP = 10.0  # prevents any single rare label from dominating loss

# ── STEP 1: Load dataset ───────────────────────────────────────────────────────
print("\n[1/6] Loading GoEmotions (simplified) dataset …")
dataset = load_dataset("google-research-datasets/go_emotions", "simplified")   # namespaced Parquet repo; the short id no longer works
print(f"      Train: {len(dataset['train']):,} | Val: {len(dataset['validation']):,} | Test: {len(dataset['test']):,}")

# ── STEP 2: Load tokenizer and model ──────────────────────────────────────────
print(f"\n[2/6] Loading tokenizer and model from '{MODEL_NAME}' …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification",
    ignore_mismatched_sizes=True,
)
# Label names come from the dataset itself (GoEmotions order, neutral = index 27).
LABEL_NAMES = dataset["train"].features["labels"].feature.names
assert len(LABEL_NAMES) == NUM_LABELS, LABEL_NAMES
model.config.id2label = {i: n for i, n in enumerate(LABEL_NAMES)}
model.config.label2id = {n: i for i, n in enumerate(LABEL_NAMES)}
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"      Device: {device} | Labels: {NUM_LABELS}")

# ── STEP 3: Tokenise ───────────────────────────────────────────────────────────
print("\n[3/6] Tokenising dataset …")
def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

tokenized = dataset.map(
    tokenize_batch, batched=True, remove_columns=["text", "id"], desc="Tokenising"
)

# ── STEP 4: Multi-hot labels ───────────────────────────────────────────────────
print("\n[4/6] Converting label lists to multi-hot float vectors …")
def make_multi_hot(batch):
    batch_size = len(batch["labels"])
    multi_hot = np.zeros((batch_size, NUM_LABELS), dtype=np.float32)
    for i, label_ids in enumerate(batch["labels"]):
        multi_hot[i, label_ids] = 1.0
    return {"labels": multi_hot.tolist()}

tokenized = tokenized.map(make_multi_hot, batched=True, desc="Building multi-hot labels")
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ── STEP 4b: Compute pos_weight ───────────────────────────────────────────────
print("\n[4b] Computing pos_weight for class-imbalance correction …")
def compute_pos_weights(dataset_split, num_labels, cap=POS_WEIGHT_CAP):
    all_labels = np.zeros((len(dataset_split), num_labels), dtype=np.float32)
    for i, example in enumerate(dataset_split):
        all_labels[i] = np.array(example["labels"])
    pos_counts = all_labels.sum(axis=0).clip(min=1)
    neg_counts = len(dataset_split) - pos_counts
    weights = neg_counts / pos_counts
    weights = np.clip(weights, a_min=None, a_max=cap)  # ← THE KEY FIX
    return torch.tensor(weights, dtype=torch.float32)

pos_weights = compute_pos_weights(tokenized["train"], NUM_LABELS)
print(f"      pos_weight range (after cap): {pos_weights.min():.1f} – {pos_weights.max():.1f}")

# ── STEP 5: Metrics with threshold sweep ──────────────────────────────────────
# CRITICAL: this must be defined BEFORE the Trainer is created
print("\n[5/6] Defining evaluation metrics …")

# Will be updated after val sweep; used at test time
BEST_THRESHOLD = 0.3

def compute_metrics(eval_pred):
    global BEST_THRESHOLD
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    probs = 1.0 / (1.0 + np.exp(-logits))
    targets = labels.astype(int)

    # Sweep thresholds 0.2 → 0.55, pick best micro-F1
    best_micro, best_thresh = 0.0, 0.3
    for thresh in np.arange(0.20, 0.75, 0.05):
        p = (probs > thresh).astype(int)
        score = f1_score(targets, p, average="micro", zero_division=0)
        if score > best_micro:
            best_micro, best_thresh = score, thresh

    # Update global so test evaluation uses the best val threshold
    BEST_THRESHOLD = best_thresh

    preds = (probs > best_thresh).astype(int)
    return {
        "micro_f1":       round(f1_score(targets, preds, average="micro", zero_division=0), 4),
        "macro_f1":       round(f1_score(targets, preds, average="macro", zero_division=0), 4),
        "best_threshold": round(best_thresh, 2),
    }

def collate_fn(batch):
    collated = default_data_collator(batch)
    collated["labels"] = collated["labels"].float()
    return collated

# ── STEP 6: Weighted Trainer ───────────────────────────────────────────────────
print("\n[6/6] Configuring Trainer and starting training …")

class WeightedBCETrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits
        pw = self._pos_weight.to(logits.device) if self._pos_weight is not None else None
        loss = nn.BCEWithLogitsLoss(pos_weight=pw)(logits, labels)
        return (loss, outputs) if return_outputs else loss

os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    learning_rate=LEARNING_RATE,
    warmup_steps=814,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_micro_f1",
    greater_is_better=True,
    logging_steps=100,
    seed=SEED,
    data_seed=SEED,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = WeightedBCETrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,   # ← defined above, not below
    data_collator=collate_fn,
    pos_weight=pos_weights,
)

print("      Training started …\n")
train_result = trainer.train()
trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)

# ── STEP 7: Save ───────────────────────────────────────────────────────────────
print(f"\n[7/7] Saving model and tokenizer to '{OUTPUT_DIR}' …")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# labels.json lets EmotionClassifier use the right label order for this checkpoint.
import json
with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as _f:
    json.dump({"labels": LABEL_NAMES, "num_labels": NUM_LABELS}, _f, indent=2)
print("      Saved successfully (labels.json written).")

# ── FIX: re-tune the threshold for the model that is actually tested ──────────
# BEST_THRESHOLD was last set by the FINAL epoch's evaluation, but
# load_best_model_at_end may have restored an earlier epoch's weights.
val_output = trainer.predict(tokenized["validation"])
_vp = 1.0 / (1.0 + np.exp(-val_output.predictions))
_vt = val_output.label_ids.astype(int)
BEST_THRESHOLD = max(
    np.arange(0.20, 0.75, 0.05),
    key=lambda th: f1_score(_vt, (_vp > th).astype(int), average="micro", zero_division=0),
)

# ── STEP 8: Test evaluation ────────────────────────────────────────────────────
print("\n[8/8] Evaluating on test set …")
test_output = trainer.predict(tokenized["test"])

# Use the best threshold found during validation
logits = test_output.predictions
labels = test_output.label_ids
probs = 1.0 / (1.0 + np.exp(-logits))
preds = (probs > BEST_THRESHOLD).astype(int)
targets = labels.astype(int)

test_metrics = {
    "micro_f1":       round(f1_score(targets, preds, average="micro", zero_division=0), 4),
    "macro_f1":       round(f1_score(targets, preds, average="macro", zero_division=0), 4),
    "threshold_used": round(BEST_THRESHOLD, 2),
}

print("\n" + "═" * 50)
print("  Final test-set results")
print("═" * 50)
print(f"  Micro-F1  : {test_metrics['micro_f1']:.4f}")
print(f"  Macro-F1  : {test_metrics['macro_f1']:.4f}")
print(f"  Threshold : {test_metrics['threshold_used']}")
print("═" * 50)

trainer.save_metrics("test", test_metrics)
print(f"\nDone. Model saved to: {os.path.abspath(OUTPUT_DIR)}")