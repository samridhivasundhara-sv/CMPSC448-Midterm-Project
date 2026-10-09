"""RQ2 follow-up: why can the prompt alone predict the LLM at all?

In Chatbot Arena the *same* prompt is answered by two different models, so a
prompt-only classifier must give both answers the same label and can be right
on at most one of them. Anything above chance therefore has to come from the
prompt *distribution* differing between families, e.g. because models were
online during different months (topics, languages and user populations drift).
"""
import re

import numpy as np
import pandas as pd

from common import LABEL2ID, LABELS, load_data, save_json, train_tfidf

NON_ASCII = re.compile(r"[^\x00-\x7f]")


def main():
    df = load_data()
    tr, te = df[df.split == "train"], df[df.split == "test"].copy()
    y = lambda d: [LABEL2ID[l] for l in d.LLM_name]
    res, vec, clf = train_tfidf(list(tr.LLM_input), y(tr), {"test": (list(te.LLM_input), y(te))})
    te["pred"] = [LABELS[i] for i in clf.predict(vec.transform(te.LLM_input))]
    te["correct"] = te.pred == te.LLM_name
    out = {"input_only_tfidf": {k: res["test"][k] for k in ("accuracy", "macro_f1")}}

    # 1) paired battles: both answers of one battle are in the test set
    paired = te.groupby("battle_id").filter(lambda g: len(g) == 2)
    out["paired"] = {
        "n_test_rows_in_pairs": int(len(paired)),
        "share_of_test": float(len(paired) / len(te)),
        "accuracy_on_paired_rows": float(paired.correct.mean()),
        "accuracy_on_unpaired_rows": float(te[~te.index.isin(paired.index)].correct.mean()),
        "max_possible_on_paired_rows": 0.5,
    }

    # 2) does the prompt population differ by family?
    df["prompt_words"] = df.LLM_input.str.split().str.len()
    df["non_english"] = df.LLM_input.map(lambda s: len(NON_ASCII.findall(s)) / max(len(s), 1) > 0.2)
    prof = df.groupby("LLM_name").agg(prompt_words_median=("prompt_words", "median"),
                                      non_english_share=("non_english", "mean"))
    prof = pd.concat([prof, pd.crosstab(df.LLM_name, df.task, normalize="index").add_prefix("task_")], axis=1)
    out["prompt_profile"] = prof.round(4).to_dict(orient="index")

    # 3) model versions per family (different release dates -> different time windows)
    out["versions"] = {f: g.model.value_counts().to_dict() for f, g in df.groupby("LLM_name")}

    # 4) prompt n-grams most associated with each family
    names = np.array(vec.get_feature_names_out())
    out["top_prompt_ngrams"] = {LABELS[i]: names[np.argsort(-clf.coef_[i])[:20]].tolist()
                                for i in range(len(LABELS))}

    save_json(out, "rq2_analysis.json")
    print(out["input_only_tfidf"], out["paired"], sep="\n")
    print(prof.round(3).to_string())
    for k, v in out["top_prompt_ngrams"].items():
        print(k, v[:12])


if __name__ == "__main__":
    main()
