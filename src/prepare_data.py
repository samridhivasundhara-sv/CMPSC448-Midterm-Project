"""Build the (LLM_name, LLM_input, LLM_output) dataset from Chatbot Arena.

Source: lmarena-ai/arena-human-preference-55k (Hugging Face). Every row is one
anonymous "battle": a real user prompt answered by two different models. We
turn each battle into up to two (family, prompt, response) examples.

Steps
  1. keep the first turn of each conversation only (prompt[0], response[0])
  2. map model versions -> 5 families (GPT, Claude, Llama, Mistral, Google)
  3. drop empty/null/very short responses
  4. tag every prompt with a coarse task type (code/math/writing/qa) for RQ3
  5. sample an equal number of examples per family
  6. split train/val/test by *prompt group* so an identical prompt never
     appears in two splits (Arena has many repeated prompts such as "test")
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/train.csv"
OUT = ROOT / "data/dataset.csv"
PER_FAMILY = 4000
SEED = 42

FAMILY_PATTERNS = [
    ("GPT", r"^gpt-"),
    ("Claude", r"^claude-"),
    ("Llama", r"^llama-2-"),  # Meta Llama-2 chat models (codellama excluded)
    ("Mistral", r"^(mistral-|mixtral-)"),
    ("Google", r"^(gemini-|palm-)"),
]


def family_of(model):
    for fam, pat in FAMILY_PATTERNS:
        if re.match(pat, model):
            return fam
    return None


def first_turn(s):
    try:
        turns = json.loads(s)
    except (TypeError, json.JSONDecodeError):
        return None
    if not turns or turns[0] is None:
        return None
    return str(turns[0]).strip()


CODE_RE = re.compile(
    r"```|\b(python|javascript|typescript|java|c\+\+|c#|rust|golang|sql|html|css|"
    r"bash|regex|code|function|script|program|compile|bug|debug|api|class|"
    r"array|algorithm|implement|json|react|django|pandas|numpy)\b", re.I)
MATH_RE = re.compile(
    r"\b(solve|calculate|equation|integral|derivative|probability|prime|"
    r"sum of|how many|math|algebra|geometry|percent|\d+\s*[\+\-\*/\^x]\s*\d+)\b", re.I)
WRITING_RE = re.compile(
    r"\b(write|compose|draft|rewrite|poem|story|essay|letter|email|song|lyrics|"
    r"haiku|limerick|joke|script for|blog|tweet|slogan|paraphrase)\b", re.I)


def task_of(prompt):
    """Keyword heuristic; order matters (code beats writing for 'write a function')."""
    if CODE_RE.search(prompt):
        return "code"
    if MATH_RE.search(prompt):
        return "math"
    if WRITING_RE.search(prompt):
        return "writing"
    return "qa"


def main():
    df = pd.read_csv(RAW)
    rows = []
    for r in df.itertuples():
        prompt = first_turn(r.prompt)
        if not prompt:
            continue
        for model, resp in ((r.model_a, r.response_a), (r.model_b, r.response_b)):
            fam = family_of(model)
            out = first_turn(resp)
            if fam is None or not out or len(out.split()) < 3:
                continue
            rows.append({"battle_id": r.id, "model": model, "family": fam,
                         "prompt": prompt, "response": out})
    data = pd.DataFrame(rows)
    data["task"] = data.prompt.map(task_of)
    print("available per family:\n", data.family.value_counts())

    data = pd.concat([g.sample(min(PER_FAMILY, len(g)), random_state=SEED)
                      for _, g in data.groupby("family")], ignore_index=True)

    # group split on normalised prompt text: 70 / 10 / 20
    key = data.prompt.str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    groups = np.array(sorted(key.unique()), dtype=object)
    rng = np.random.default_rng(SEED)
    rng.shuffle(groups)
    n = len(groups)
    split_of = {g: "train" for g in groups[: int(0.7 * n)]}
    split_of.update({g: "val" for g in groups[int(0.7 * n): int(0.8 * n)]})
    split_of.update({g: "test" for g in groups[int(0.8 * n):]})
    data["split"] = key.map(split_of)

    data = data.rename(columns={"family": "LLM_name", "prompt": "LLM_input",
                                "response": "LLM_output"})
    data = data[["LLM_name", "model", "LLM_input", "LLM_output", "task", "split", "battle_id"]]
    data.to_csv(OUT, index=False)
    print(f"saved {len(data)} rows to {OUT}")
    print(pd.crosstab(data.LLM_name, data.split))
    print(pd.crosstab(data.LLM_name, data.task))


if __name__ == "__main__":
    main()
