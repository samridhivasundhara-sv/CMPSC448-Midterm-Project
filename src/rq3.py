"""RQ3: do LLM fingerprints transfer across task types?

Two cross-task settings (both use the response text only, as in RQ1):
  A. train on code + math + qa      -> test on writing
  B. train on qa (general questions) -> test on code

For each setting we report
  * in-domain:     test examples from the *training* task types
  * cross-domain:  test examples from the held-out task type
The RQ1 models (trained on all tasks) evaluated on the same held-out task are
the "seen-task" reference; they are read from rq1_rq2 results when plotting.
"""
import sys

import numpy as np

from common import (LABEL2ID, load_data, make_texts, make_tokens, save_json,
                    train_neural, train_tfidf)

SETTINGS = {
    "A_to_writing": (["code", "math", "qa"], "writing"),
    "B_qa_to_code": (["qa"], "code"),
}
SEEDS = [0, 1, 2]
KINDS = sys.argv[1:] or ["tfidf", "cnn", "lstm"]
VIEW = "output"


def main():
    df = load_data()
    y = lambda d: [LABEL2ID[l] for l in d.LLM_name]
    results = {}
    for name, (src_tasks, tgt) in SETTINGS.items():
        src = df[df.task.isin(src_tasks)]
        tr, va = src[src.split == "train"], src[src.split == "val"]
        te_in = src[src.split == "test"]
        te_out = df[(df.task == tgt) & (df.split == "test")]
        print(f"### {name}: train {len(tr)} | in-domain test {len(te_in)} | "
              f"cross-domain test {len(te_out)}", flush=True)
        for kind in KINDS:
            runs = []
            if kind == "tfidf":
                tests = {"in_domain": (make_texts(te_in, VIEW), y(te_in)),
                         "cross_domain": (make_texts(te_out, VIEW), y(te_out))}
                runs.append(train_tfidf(make_texts(tr, VIEW), y(tr), tests)[0])
            else:
                tests = {"in_domain": (make_tokens(te_in, VIEW), y(te_in)),
                         "cross_domain": (make_tokens(te_out, VIEW), y(te_out))}
                for seed in SEEDS:
                    runs.append(train_neural(kind, make_tokens(tr, VIEW), y(tr),
                                             make_tokens(va, VIEW), y(va), tests,
                                             seed=seed, verbose=False))
            entry = {"n_train": len(tr), "runs": runs}
            for split in ("in_domain", "cross_domain"):
                for m in ("accuracy", "macro_f1"):
                    v = [r[split][m] for r in runs]
                    entry[f"{split}_{m}_mean"] = float(np.mean(v))
                    entry[f"{split}_{m}_std"] = float(np.std(v))
            results[f"{name}/{kind}"] = entry
            print(f"== {name} {kind:5s} in-domain F1 {entry['in_domain_macro_f1_mean']:.3f} | "
                  f"cross-domain F1 {entry['cross_domain_macro_f1_mean']:.3f}", flush=True)
            save_json(results, "rq3.json")


if __name__ == "__main__":
    main()
