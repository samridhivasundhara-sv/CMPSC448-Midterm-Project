# Who Wrote It? Identifying LLMs from Their Responses

CMPSC 448 midterm project. We train a **CNN** and a **bidirectional LSTM** (plus a
TF-IDF + logistic-regression reference) to predict which LLM family — **Claude,
GPT, Google (Gemini/PaLM), Llama or Mistral** — wrote a response, and study
RQ1–RQ4 from the assignment.

The full write-up is in [`report/report.pdf`](report/report.pdf).

## Data

* Source: [`lmarena-ai/arena-human-preference-55k`](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k)
  (Chatbot Arena, public on Hugging Face). Each row is a real user prompt answered
  anonymously by two different models.
* `src/prepare_data.py` turns it into `data/dataset.csv` with the columns
  `LLM_name` (family), `model` (exact version), `LLM_input` (prompt),
  `LLM_output` (response), `task` (code / math / writing / qa, keyword heuristic),
  `split` (train / val / test) and `battle_id`.
* 4,000 examples per family (20,000 in total), first turn of each conversation only.
* The split is 70 / 10 / 20 **by prompt**: identical prompts never cross splits.

## Layout

```
src/prepare_data.py   build data/dataset.csv from the raw Arena file
src/common.py         tokenizer, vocabulary, TextCNN, BiLSTM, training loop, TF-IDF baseline
src/rq1_rq2.py        RQ1 + RQ2: input-only / output-only / input+output, 3 seeds
src/rq2_analysis.py   RQ2 follow-up: why prompt-only is (not) predictive
src/rq3.py            RQ3: train on some task types, test on an unseen one
src/rq4.py            RQ4: stylometric features, feature-only classifier, top n-grams, ablations
src/plots.py          all figures -> figures/
results/              JSON / CSV results of every experiment, tuning notes
logs/                 training logs
report/               report source (HTML) and PDF
run_all.sh            reproduce everything
```

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('lmarena-ai/arena-human-preference-55k','train.csv',repo_type='dataset',local_dir='data/raw')"
./run_all.sh          # about 2-3 hours on an Apple M2 Pro (MPS); also runs on CUDA or CPU
```

## Team

* Team leader: Samridhi Vasundhara
* Members: Samridhi Vasundhara, Rishita Shukla, Harshitha Rao Renati
