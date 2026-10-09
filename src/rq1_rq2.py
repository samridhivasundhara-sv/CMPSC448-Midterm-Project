"""RQ1 (output only) and RQ2 (input only vs output only vs input + output).

For each view we train the CNN and the BiLSTM with 3 seeds, plus a TF-IDF +
logistic-regression reference. Test metrics are also broken down by task type,
which RQ3 re-uses as the "trained on everything" reference.
"""
import sys

import numpy as np

from common import (LABEL2ID, load_data, make_texts, make_tokens, save_json,
                    train_neural, train_tfidf)

SEEDS = [0, 1, 2]
VIEWS = ["output", "input", "both"]
KINDS = sys.argv[1:] or ["tfidf", "cnn", "lstm"]


def main():
    df = load_data()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    y = lambda d: [LABEL2ID[l] for l in d.LLM_name]
    tasks = sorted(df.task.unique())

    results = {}
    for view in VIEWS:
        for kind in KINDS:
            runs = []
            if kind == "tfidf":
                tests = {"test": (make_texts(te, view), y(te))}
                tests.update({f"test_{t}": (make_texts(te[te.task == t], view), y(te[te.task == t]))
                              for t in tasks})
                runs.append(train_tfidf(make_texts(tr, view), y(tr), tests)[0])
            else:
                tests = {"test": (make_tokens(te, view), y(te))}
                tests.update({f"test_{t}": (make_tokens(te[te.task == t], view), y(te[te.task == t]))
                              for t in tasks})
                for seed in SEEDS:
                    runs.append(train_neural(kind, make_tokens(tr, view), y(tr),
                                             make_tokens(va, view), y(va), tests, seed=seed))
            accs = [r["test"]["accuracy"] for r in runs]
            f1s = [r["test"]["macro_f1"] for r in runs]
            results[f"{kind}/{view}"] = {
                "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
                "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                "runs": runs,
            }
            print(f"== {kind:5s} {view:6s} acc {np.mean(accs):.3f}±{np.std(accs):.3f} "
                  f"macroF1 {np.mean(f1s):.3f}±{np.std(f1s):.3f}", flush=True)
            save_json(results, f"rq1_rq2_{'_'.join(KINDS)}.json")


if __name__ == "__main__":
    main()
