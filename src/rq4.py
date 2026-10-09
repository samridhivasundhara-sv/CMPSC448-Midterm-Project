"""RQ4: what distinguishes the LLM families?

Part 1  Hand-crafted stylometric features (length, lexical diversity,
        formatting, punctuation, phrasing) -> per-family statistics.
Part 2  A classifier that sees ONLY those ~25 numbers (no words), plus
        permutation importance to rank them.
Part 3  Most indicative n-grams per family from the TF-IDF model.
Part 4  Ablations: re-train CNN/BiLSTM after removing one kind of signal
        from the responses and measure the drop.
"""
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

from common import (LABEL2ID, LABELS, RESULTS, load_data, save_json, tokenize,
                    train_neural, train_tfidf)

# ------------------------------------------------------------ Part 1 features
FEATURE_GROUPS = {
    "length": ["n_words", "n_chars", "n_sentences", "avg_sentence_len", "avg_word_len"],
    "lexical": ["ttr_first100", "hapax_ratio", "first_person_rate", "you_rate",
                "uppercase_ratio", "digit_ratio"],
    "formatting": ["n_lines", "n_paragraphs", "n_bullets", "n_numbered", "n_headers",
                   "n_bold", "n_code_blocks", "has_colon_intro"],
    "punctuation": ["comma_rate", "exclam_rate", "question_rate", "colon_rate",
                    "paren_rate", "emoji_count"],
    "phrasing": ["starts_affirmation", "apology_or_refusal", "mentions_ai",
                 "ends_with_offer"],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]

WORD_RE = re.compile(r"[A-Za-z']+")
SENT_RE = re.compile(r"[.!?]+(?:\s|$)")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
AFFIRM_RE = re.compile(r"^(sure|certainly|of course|absolutely|great question|yes|okay|ok)\b", re.I)
SORRY_RE = re.compile(r"\b(i'?m sorry|i apologi[sz]e|i cannot|i can'?t|i am unable|i'?m unable|"
                      r"i'?m not able|as an ai)\b", re.I)
AI_RE = re.compile(r"\b(ai|language model|assistant|chatbot)\b", re.I)
OFFER_RE = re.compile(r"(let me know|feel free|hope (this|that) helps|anything else)[^\n]*$", re.I)


def features(text):
    words = WORD_RE.findall(text)
    nw = max(len(words), 1)
    lower = [w.lower() for w in words]
    counts = Counter(lower)
    lines = text.split("\n")
    n_sent = max(len(SENT_RE.findall(text)), 1)
    chars = max(len(text), 1)
    return {
        "n_words": len(words),
        "n_chars": len(text),
        "n_sentences": n_sent,
        "avg_sentence_len": len(words) / n_sent,
        "avg_word_len": sum(map(len, words)) / nw,
        "ttr_first100": len(set(lower[:100])) / max(min(len(lower), 100), 1),
        "hapax_ratio": sum(1 for c in counts.values() if c == 1) / nw,
        "first_person_rate": sum(counts[w] for w in ("i", "me", "my", "i'm", "i've")) / nw,
        "you_rate": sum(counts[w] for w in ("you", "your", "you're")) / nw,
        "uppercase_ratio": sum(c.isupper() for c in text) / chars,
        "digit_ratio": sum(c.isdigit() for c in text) / chars,
        "n_lines": len(lines),
        "n_paragraphs": text.count("\n\n") + 1,
        "n_bullets": sum(bool(re.match(r"\s*[-*•]\s", l)) for l in lines),
        "n_numbered": sum(bool(re.match(r"\s*\d+[.)]\s", l)) for l in lines),
        "n_headers": sum(bool(re.match(r"\s*#{1,6}\s", l)) for l in lines),
        "n_bold": text.count("**") // 2,
        "n_code_blocks": text.count("```") // 2,
        "has_colon_intro": int(":" in lines[0][-3:]) if lines else 0,
        "comma_rate": text.count(",") / nw,
        "exclam_rate": text.count("!") / nw,
        "question_rate": text.count("?") / nw,
        "colon_rate": text.count(":") / nw,
        "paren_rate": text.count("(") / nw,
        "emoji_count": len(EMOJI_RE.findall(text)),
        "starts_affirmation": int(bool(AFFIRM_RE.match(text.strip()))),
        "apology_or_refusal": int(bool(SORRY_RE.search(text))),
        "mentions_ai": int(bool(AI_RE.search(text))),
        "ends_with_offer": int(bool(OFFER_RE.search(text.strip()[-200:]))),
    }


SELF_ID = {"OpenAI": r"\bopenai\b", "ChatGPT/GPT": r"\b(chatgpt|gpt-?\d)",
           "Anthropic": r"\banthropic\b", "Claude": r"\bclaude\b",
           "Google": r"\bgoogle\b", "Meta": r"\bmeta\b", "Llama": r"\bllama\b",
           "Mistral": r"\bmistral\b"}


# ------------------------------------------------------------ Part 4 ablations
def strip_formatting(t):
    """Remove markdown / layout: bullets, headers, bold, code fences, newlines."""
    t = re.sub(r"```.*?\n", " ", t)
    t = re.sub(r"^\s*(#{1,6}|[-*•]|\d+[.)])\s+", "", t, flags=re.M)
    t = t.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"\s+", " ", t).strip()


def words_only(t):
    """Lower-case words only: removes punctuation, case, layout, emoji."""
    return " ".join(w.lower() for w in WORD_RE.findall(t))


def first_50(t):
    """Keep only the first 50 tokens: removes (most of) the length signal."""
    return " ".join(tokenize(t)[:50]).replace("<nl>", "\n")


def make_function_word_mask(train_texts, keep=300):
    """Keep the `keep` most frequent tokens of the training set (function words,
    punctuation, formatting symbols); every other word becomes <w>. Removes
    topic/content while preserving style."""
    counts = Counter(t.lower() for x in train_texts for t in tokenize(x))
    vocab = {w for w, _ in counts.most_common(keep)}
    return lambda t: " ".join(w if w.lower() in vocab else "<w>" for w in tokenize(t)).replace(
        "<nl>", "\n")


def shuffle_words(seed=0):
    """Randomly permute tokens inside each response: keeps the bag of words,
    destroys word order."""
    rng = np.random.default_rng(seed)

    def f(t):
        toks = tokenize(t)[:384]
        rng.shuffle(toks)
        return " ".join(toks).replace("<nl>", "\n")
    return f


ABL_SEEDS = {"cnn": [0, 1, 2], "lstm": [0]}


def main():
    df = load_data()
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    y = lambda d: [LABEL2ID[l] for l in d.LLM_name]
    out = {}
    parts = sys.argv[1:] or ["1", "2", "3", "4"]

    # ---- Part 1
    feats = pd.DataFrame([features(t) for t in df.LLM_output])
    feats["family"] = df.LLM_name.values
    feats.to_csv(RESULTS / "rq4_features.csv", index=False)
    if "1" in parts:
        stats = feats.groupby("family")[FEATURES].agg(["mean", "median"])
        stats.to_csv(RESULTS / "rq4_feature_stats.csv")
        print(feats.groupby("family")[FEATURES].mean().T.round(3).to_string())
        selfid = {fam: {k: float(df[df.LLM_name == fam].LLM_output.str.contains(p, case=False, regex=True).mean())
                        for k, p in SELF_ID.items()} for fam in LABELS}
        out["self_identification_rate"] = selfid
        print(pd.DataFrame(selfid).round(3).to_string())

    # ---- Part 2
    if "2" in parts:
        Xtr, Xte = feats.loc[tr.index, FEATURES], feats.loc[te.index, FEATURES]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        clf.fit(Xtr, y(tr))
        pred = clf.predict(Xte)
        res = {"accuracy": float((pred == np.array(y(te))).mean()),
               "macro_f1": float(f1_score(y(te), pred, average="macro"))}
        imp = permutation_importance(clf, Xte, y(te), scoring="f1_macro", n_repeats=5, random_state=0)
        res["permutation_importance"] = dict(sorted(zip(FEATURES, imp.importances_mean.tolist()),
                                                    key=lambda kv: -kv[1]))
        # one group at a time: which kind of feature carries the signal?
        res["group_only"] = {}
        for g, cols in FEATURE_GROUPS.items():
            c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
            c.fit(feats.loc[tr.index, cols], y(tr))
            p = c.predict(feats.loc[te.index, cols])
            res["group_only"][g] = float(f1_score(y(te), p, average="macro"))
        out["feature_classifier"] = res
        print("feature-only classifier:", {k: v for k, v in res.items() if k != "permutation_importance"})
        print("top features:", list(res["permutation_importance"].items())[:10])

    # ---- Part 3
    if "3" in parts:
        _, vec, clf = train_tfidf(list(tr.LLM_output), y(tr), {})
        names = np.array(vec.get_feature_names_out())
        out["top_ngrams"] = {LABELS[i]: names[np.argsort(-clf.coef_[i])[:25]].tolist()
                             for i in range(len(LABELS))}
        for fam, ngr in out["top_ngrams"].items():
            print(fam, ngr[:15])

    # ---- Part 4
    if "4" in parts:
        fw = make_function_word_mask(tr.LLM_output)
        ablations = {
            "original": lambda t: t,
            "no_formatting": strip_formatting,
            "words_only": words_only,
            "first_50_tokens": first_50,
            "content_masked": fw,
            "shuffled_order": shuffle_words(0),
        }
        out["ablation"] = {}
        for name, fn in ablations.items():
            toks = {s: [tokenize(fn(t))[:384] for t in d.LLM_output]
                    for s, d in (("tr", tr), ("va", va), ("te", te))}
            for kind, seeds in ABL_SEEDS.items():
                f1s = [train_neural(kind, toks["tr"], y(tr), toks["va"], y(va),
                                    {"test": (toks["te"], y(te))}, seed=s,
                                    verbose=False)["test"]["macro_f1"] for s in seeds]
                out["ablation"][f"{name}/{kind}"] = {"f1_mean": float(np.mean(f1s)),
                                                     "f1_std": float(np.std(f1s)), "f1s": f1s}
                print(f"== ablation {name:16s} {kind:5s} macroF1 {np.mean(f1s):.3f}", flush=True)
                save_json(out, f"rq4_part{''.join(parts)}.json")
        print("example (no_formatting):", strip_formatting(te.LLM_output.iloc[0])[:200])
        print("example (content_masked):", fw(te.LLM_output.iloc[0])[:200])

    save_json(out, f"rq4_part{''.join(parts)}.json")


if __name__ == "__main__":
    main()
