"""
Fine-tune BERT on the Jigsaw Toxic Comment Classification Dataset
Summer Internship Project | Summer of AI

Usage:
  python scripts/train.py \
    --data data/train.csv \
    --output models/bert-toxic-finetuned \
    --epochs 3 \
    --batch_size 16
"""

import argparse
import logging
import os
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def load_data(csv_path: str):
    """Load and validate Jigsaw CSV dataset."""
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} rows")
    required = ["comment_text"] + LABEL_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=["comment_text"])
    return df


def train(args):
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        from torch.optim import AdamW
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Run: pip install torch transformers scikit-learn pandas")
        return

    df = load_data(args.data)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    class JigsawDataset(Dataset):
        def __init__(self, dataframe):
            self.texts = dataframe["comment_text"].tolist()
            self.labels = dataframe[LABEL_COLS].values.astype(np.float32)

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            enc = tokenizer(
                self.texts[idx],
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(self.labels[idx]),
            }

    train_loader = DataLoader(JigsawDataset(train_df), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(JigsawDataset(val_df),   batch_size=args.batch_size * 2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_auc = 0
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        # ── Training ──────────────────────────────────────────
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            ids   = batch["input_ids"].to(device)
            mask  = batch["attention_mask"].to(device)
            lbls  = batch["labels"].to(device)

            outputs = model(input_ids=ids, attention_mask=mask)
            loss = loss_fn(outputs.logits, lbls)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 100 == 0:
                logger.info(f"Epoch {epoch+1} step {step}/{len(train_loader)} | loss {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} avg train loss: {avg_loss:.4f}")

        # ── Validation ────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids   = batch["input_ids"].to(device)
                mask  = batch["attention_mask"].to(device)
                lbls  = batch["labels"]
                out   = model(input_ids=ids, attention_mask=mask)
                preds = torch.sigmoid(out.logits).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(lbls.numpy())

        all_preds  = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        auc = roc_auc_score(all_labels, all_preds, average="macro")
        logger.info(f"Epoch {epoch+1} val ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            os.makedirs(args.output, exist_ok=True)
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)
            logger.info(f"✅ Saved best model to {args.output} (AUC={best_auc:.4f})")

    logger.info(f"Training complete. Best ROC-AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for toxic comment detection")
    parser.add_argument("--data",       default="data/train.csv",                    help="Path to Jigsaw CSV")
    parser.add_argument("--model_name", default="bert-base-uncased",                 help="Pretrained model")
    parser.add_argument("--output",     default="models/bert-toxic-finetuned",       help="Output directory")
    parser.add_argument("--epochs",     type=int,   default=3,                        help="Training epochs")
    parser.add_argument("--batch_size", type=int,   default=16,                       help="Batch size")
    parser.add_argument("--lr",         type=float, default=2e-5,                     help="Learning rate")
    args = parser.parse_args()
    train(args)
