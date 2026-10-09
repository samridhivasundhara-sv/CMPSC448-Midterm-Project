"""Shared pieces: tokenizer, vocabulary, CNN / BiLSTM models, training + evaluation."""
import gc
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/dataset.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

LABELS = ["Claude", "GPT", "Google", "Llama", "Mistral"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

PAD, UNK, SEP, NL = "<pad>", "<unk>", "<sep>", "<nl>"

# Newlines are kept as a token on purpose: line breaks, bullets ("-", "*", "1."),
# markdown ("**", "#") and punctuation are part of an LLM's style.
TOKEN_RE = re.compile(r"\n|\w+|[^\w\s]")

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    return pd.read_csv(DATA, keep_default_na=False)


def tokenize(text):
    return [NL if t == "\n" else t for t in TOKEN_RE.findall(text)]


# ---------------------------------------------------------------- input views
# max tokens kept for each part of the input
MAX_IN, MAX_OUT = 128, 384


def make_tokens(df, view):
    """view: 'input' (prompt only), 'output' (response only) or 'both'."""
    if view == "input":
        return [tokenize(p)[:MAX_IN] for p in df.LLM_input]
    if view == "output":
        return [tokenize(r)[:MAX_OUT] for r in df.LLM_output]
    if view == "both":
        return [tokenize(p)[:MAX_IN] + [SEP] + tokenize(r)[:MAX_OUT]
                for p, r in zip(df.LLM_input, df.LLM_output)]
    raise ValueError(view)


def make_texts(df, view):
    """Plain-string version of each view for the TF-IDF baseline."""
    if view == "input":
        return list(df.LLM_input)
    if view == "output":
        return list(df.LLM_output)
    return [p + " <sep> " + r for p, r in zip(df.LLM_input, df.LLM_output)]


class Vocab:
    def __init__(self, token_lists, min_freq=2, max_size=30000):
        counts = Counter(t for toks in token_lists for t in toks)
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        self.itos = [PAD, UNK, SEP] + [w for w in words if w not in (PAD, UNK, SEP)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi.get(t, 1) for t in toks] or [1]


def batches(ids, labels, batch_size, shuffle):
    """Bucketed batching: sequences of similar length go together (less padding).
    When shuffling, we shuffle, sort within chunks of 50 batches by length,
    then shuffle the order of the batches."""
    n = len(ids)
    if shuffle:
        order = np.random.permutation(n)
        chunk = batch_size * 50
        order = np.concatenate([sorted(order[i:i + chunk], key=lambda j: len(ids[j]))
                                for i in range(0, n, chunk)])
        starts = np.random.permutation(range(0, n, batch_size))
    else:
        order = np.argsort([len(x) for x in ids], kind="stable")
        starts = range(0, n, batch_size)
    for s in starts:
        idx = order[s:s + batch_size]
        seqs = [ids[i] for i in idx]
        lens = torch.tensor([len(x) for x in seqs])
        # round the padded length up to a multiple of 32: far fewer distinct
        # tensor shapes, which keeps the MPS kernel/memory cache small
        T = -(-int(max(lens)) // 32) * 32
        x = torch.zeros(len(seqs), T, dtype=torch.long)
        for j, seq in enumerate(seqs):
            x[j, :len(seq)] = torch.tensor(seq)
        yield x, lens, torch.tensor([labels[i] for i in idx])


# --------------------------------------------------------------------- models
class TextCNN(nn.Module):
    """Kim (2014)-style CNN: parallel 1-D convolutions of widths 3/4/5 over word
    embeddings, max-pooled over time, then a linear classifier."""

    def __init__(self, vocab_size, n_classes, emb_dim=128, n_filters=128,
                 widths=(3, 4, 5), dropout=0.5, emb_dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.convs = nn.ModuleList(nn.Conv1d(emb_dim, n_filters, w, padding=w // 2)
                                   for w in widths)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(widths), n_classes)

    def forward(self, x, lens):
        mask = (x != 0).unsqueeze(1)                       # (B, 1, T)
        e = self.emb_drop(self.emb(x)).transpose(1, 2)         # (B, E, T)
        pooled = []
        for conv in self.convs:
            h = F.relu(conv(e))[:, :, :x.size(1)]
            h = h.masked_fill(~mask, -1e4)                 # ignore padding in max-pool
            pooled.append(h.max(dim=2).values)
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


def reverse_padded(h, lens):
    """Reverse each sequence inside its true length; padding stays at the end."""
    T = h.size(1)
    t = torch.arange(T, device=h.device).unsqueeze(0)            # (1, T)
    L = lens.to(h.device).unsqueeze(1)                            # (B, 1)
    idx = torch.where(t < L, L - 1 - t, t)                        # (B, T)
    return h.gather(1, idx.unsqueeze(2).expand_as(h))


class BiLSTM(nn.Module):
    """2-layer bidirectional LSTM; the hidden states are summarised with
    masked mean-pooling and max-pooling, then a linear classifier.

    The backward direction is computed by reversing every sequence within its
    own length, which is equivalent to PyTorch's packed bidirectional LSTM but
    ~5x faster on Apple-silicon GPUs (packed sequences are slow there).
    """

    def __init__(self, vocab_size, n_classes, emb_dim=128, hidden=128,
                 layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        dims = [emb_dim] + [2 * hidden] * (layers - 1)
        self.fwd = nn.ModuleList(nn.LSTM(d, hidden, batch_first=True) for d in dims)
        self.bwd = nn.ModuleList(nn.LSTM(d, hidden, batch_first=True) for d in dims)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, n_classes)

    def forward(self, x, lens):
        h = self.drop(self.emb(x))
        for i, (f, b) in enumerate(zip(self.fwd, self.bwd)):
            hf, _ = f(h)
            hb, _ = b(reverse_padded(h, lens))
            h = torch.cat([hf, reverse_padded(hb, lens)], dim=2)
            if i < len(self.fwd) - 1:
                h = self.drop(h)
        mask = (x != 0).unsqueeze(2).float()
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = h.masked_fill(mask == 0, -1e4).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], dim=1)))


MODELS = {"cnn": TextCNN, "lstm": BiLSTM}
HPARAMS = {  # chosen on the validation set, see results/tuning.txt
    "cnn": dict(lr=1e-3, epochs=15, batch_size=64, patience=4),
    "lstm": dict(lr=1e-3, epochs=15, batch_size=64, patience=4),
}


def predict(model, ids, labels, batch_size=128):
    model.eval()
    order = np.argsort([len(x) for x in ids], kind="stable")
    preds = []
    with torch.no_grad():
        for x, lens, _ in batches(ids, labels, batch_size, shuffle=False):
            preds.append(model(x.to(DEVICE), lens).argmax(1).cpu())
    out = np.empty(len(ids), dtype=int)
    out[order] = torch.cat(preds).numpy()
    return out


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_f1": dict(zip(LABELS, f1_score(y_true, y_pred, average=None,
                                                  labels=range(len(LABELS))).tolist())),
        "confusion": confusion_matrix(y_true, y_pred, labels=range(len(LABELS))).tolist(),
    }


def train_neural(kind, train_toks, y_train, val_toks, y_val, test_sets, seed=0,
                 verbose=True):
    """Train a CNN or BiLSTM; early-stop on validation macro-F1.

    test_sets: dict name -> (token_lists, labels). Returns dict name -> metrics,
    plus the training history.
    """
    set_seed(seed)
    hp = HPARAMS[kind]
    vocab = Vocab(train_toks)
    enc = lambda toks: [vocab.encode(t) for t in toks]
    tr, va = enc(train_toks), enc(val_toks)
    model = MODELS[kind](len(vocab), len(LABELS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_f1, best_state, bad, history = -1, None, 0, []
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, lens, y in batches(tr, y_train, hp["batch_size"], shuffle=True):
            opt.zero_grad()
            loss = loss_fn(model(x.to(DEVICE), lens), y.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)
        val_f1 = f1_score(y_val, predict(model, va, y_val), average="macro")
        history.append({"epoch": epoch, "train_loss": total / len(tr), "val_macro_f1": val_f1})
        if verbose:
            print(f"  [{kind} seed={seed}] epoch {epoch} loss {total/len(tr):.3f} "
                  f"val_f1 {val_f1:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    out = {name: metrics(y, predict(model, enc(toks), y)) for name, (toks, y) in test_sets.items()}
    out["history"] = history
    out["best_val_macro_f1"] = best_f1
    del model, opt, best_state
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    return out


def train_tfidf(train_texts, y_train, test_sets):
    """Non-neural reference point: TF-IDF word 1-2 grams + logistic regression."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000,
                          sublinear_tf=True, lowercase=False,
                          token_pattern=r"(?u)\b\w+\b|[^\w\s]")
    X = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=2000, C=5.0).fit(X, y_train)
    out = {name: metrics(y, clf.predict(vec.transform(texts)))
           for name, (texts, y) in test_sets.items()}
    return out, vec, clf


def save_json(obj, name):
    with open(RESULTS / name, "w") as f:
        json.dump(obj, f, indent=2)
