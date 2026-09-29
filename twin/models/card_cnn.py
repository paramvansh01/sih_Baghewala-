"""Dynamometer-card diagnosis: 1-D CNN (PyTorch) trained FROM SCRATCH (random init, no
pre-trained weights) on the labelled cards from the finite-difference rod/pump model.

Input: 2 x 128 sequence over one stroke (position/stroke, load/rod weight in air), as a
position transducer + load cell record it. Convolutions use circular padding because a
stroke is periodic. Split: stratified train/val/test; early stopping on validation loss.
Baselines reported: multinomial logistic regression on the same raw input, and (for the
rod_floating class) the physics rule 'minimum measured load <= threshold'.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from ..config import ROOT, load_pipeline
from ..data.generate_cards import CLASSES

DATA = ROOT / "artifacts" / "data"
MODELS = ROOT / "artifacts" / "models"
METRICS = ROOT / "artifacts" / "metrics"


class CardCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES), width=32, dropout=0.2):
        super().__init__()

        def block(cin, cout, k, stride):
            return nn.Sequential(nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2,
                                           padding_mode="circular"),
                                 nn.BatchNorm1d(cout), nn.ReLU())

        self.features = nn.Sequential(block(2, width, 7, 1), block(width, 2 * width, 5, 2),
                                      block(2 * width, 4 * width, 5, 2), block(4 * width, 4 * width, 3, 2))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width, n_classes))

    def forward(self, x):
        return self.head(self.features(x).mean(dim=-1))


def load_cards():
    meta = pd.read_csv(DATA / "cards_meta.csv")
    x = np.load(DATA / "cards_x.npy").astype(np.float32)
    return meta, x


def splits(meta, cfg, seed):
    idx = np.arange(len(meta))
    y = meta["label"].to_numpy()
    tr, te = train_test_split(idx, test_size=cfg["test_fraction"], stratify=y, random_state=seed)
    tr, va = train_test_split(tr, test_size=cfg["val_fraction"], stratify=y[tr], random_state=seed)
    return tr, va, te


def _batches(n, bs, rng):
    order = rng.permutation(n)
    for i in range(0, n, bs):
        yield order[i:i + bs]


def train(save=True, verbose=True):
    pl = load_pipeline()
    cfg, seed = pl["card_cnn"], pl["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    meta, x = load_cards()
    y = meta["label"].to_numpy()
    tr, va, te = splits(meta, cfg, seed)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y).long()
    model = CardCNN()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)

    def evaluate(ids):
        model.eval()
        with torch.no_grad():
            logits = model(xt[ids])
            return float(loss_fn(logits, yt[ids])), logits.argmax(1).numpy()

    init_val_loss, _ = evaluate(va)
    history, best, best_state, bad = [], np.inf, None, 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        tot = 0.0
        for b in _batches(len(tr), cfg["batch_size"], rng):
            ids = tr[b]
            opt.zero_grad()
            loss = loss_fn(model(xt[ids]), yt[ids])
            loss.backward()
            opt.step()
            tot += loss.item() * len(ids)
        sched.step()
        vl, vp = evaluate(va)
        history.append(dict(epoch=ep + 1, train_loss=tot / len(tr), val_loss=vl,
                            val_acc=float((vp == y[va]).mean())))
        if verbose:
            print(f"epoch {ep + 1:3d} train {tot / len(tr):.4f} val {vl:.4f} acc {history[-1]['val_acc']:.4f}")
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    _, pred = evaluate(te)
    y_te = y[te]
    report = classification_report(y_te, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, pred).tolist()
    # baseline 1: logistic regression on the raw flattened input
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    lr.fit(x[tr].reshape(len(tr), -1), y[tr])
    lr_pred = lr.predict(x[te].reshape(len(te), -1))
    # baseline 2: physics rule for floating = measured minimum load near zero (carrier-bar separation)
    min_load = x[:, 1, :].min(axis=1)
    fl = CLASSES.index("rod_floating")
    thr_grid = np.linspace(0.0, 0.3, 61)
    f1s = [f1_score(y[tr] == fl, min_load[tr] <= t) for t in thr_grid]
    thr = float(thr_grid[int(np.argmax(f1s))])
    rule_f1 = float(f1_score(y_te == fl, min_load[te] <= thr))
    results = dict(
        n_train=len(tr), n_val=len(va), n_test=len(te), n_parameters=int(n_params),
        epochs_run=len(history), train_seconds=time.time() - t0,
        initial_val_loss=init_val_loss, chance_loss=float(np.log(len(CLASSES))),
        best_val_loss=best,
        cnn=dict(accuracy=float((pred == y_te).mean()), macro_f1=float(f1_score(y_te, pred, average="macro")),
                 per_class={c: report[c] for c in CLASSES}, confusion_matrix=cm),
        logistic_regression=dict(accuracy=float((lr_pred == y_te).mean()),
                                 macro_f1=float(f1_score(y_te, lr_pred, average="macro"))),
        floating_rule=dict(threshold_load_over_w_air=thr, f1=rule_f1),
        cnn_floating_f1=float(report["rod_floating"]["f1-score"]),
        history=history, classes=CLASSES, config=cfg, seed=seed,
    )
    robust = DATA / "cards_robust_meta.csv"
    if robust.exists():  # independent seed, 3x sensor noise (see pipeline.yaml card_cnn.robustness)
        rm = pd.read_csv(robust)
        rx = np.load(DATA / "cards_robust_x.npy").astype(np.float32)
        model.eval()
        with torch.no_grad():
            rp = model(torch.from_numpy(rx)).argmax(1).numpy()
        results["robustness"] = dict(
            n=len(rm), accuracy=float((rp == rm["label"].to_numpy()).mean()),
            macro_f1=float(f1_score(rm["label"], rp, average="macro")),
            confusion_matrix=confusion_matrix(rm["label"], rp).tolist(),
            logistic_regression_accuracy=float((lr.predict(rx.reshape(len(rx), -1)) == rm["label"]).mean()))
    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        METRICS.mkdir(parents=True, exist_ok=True)
        torch.save(dict(state_dict=model.state_dict(), classes=CLASSES), MODELS / "card_cnn.pt")
        with open(METRICS / "card_cnn.json", "w") as f:
            json.dump(results, f, indent=2)
    return model, results


class CardDiagnoser:
    def __init__(self, path=None):
        b = torch.load(path or MODELS / "card_cnn.pt", weights_only=True)
        self.classes = b["classes"]
        self.model = CardCNN(len(self.classes))
        self.model.load_state_dict(b["state_dict"])
        self.model.eval()

    def predict_proba(self, cards: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(self.model(torch.from_numpy(cards.astype(np.float32))), 1).numpy()


if __name__ == "__main__":
    _, r = train()
    print(f"\nparams {r['n_parameters']}, epochs {r['epochs_run']}, initial val loss {r['initial_val_loss']:.3f} "
          f"(chance {r['chance_loss']:.3f}) -> best {r['best_val_loss']:.4f}")
    print("CNN test accuracy", round(r["cnn"]["accuracy"], 4), "macro-F1", round(r["cnn"]["macro_f1"], 4))
    print("LogReg test accuracy", round(r["logistic_regression"]["accuracy"], 4),
          "macro-F1", round(r["logistic_regression"]["macro_f1"], 4))
    print("floating F1: CNN", round(r["cnn_floating_f1"], 4), "| min-load rule", round(r["floating_rule"]["f1"], 4),
          "thr", r["floating_rule"]["threshold_load_over_w_air"])
    print(pd.DataFrame({c: r["cnn"]["per_class"][c] for c in r["classes"]}).T.round(3))
    print(pd.DataFrame(r["cnn"]["confusion_matrix"], index=r["classes"], columns=r["classes"]).to_string())
    if "robustness" in r:
        print("robustness set (new seed, 3x noise): CNN acc", round(r["robustness"]["accuracy"], 4),
              "LogReg acc", round(r["robustness"]["logistic_regression_accuracy"], 4))
