#!/bin/bash
# Reproduce every experiment end-to-end (about 2-3 h on an Apple M2 Pro).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p logs
$PY src/prepare_data.py            > logs/prepare_data.log 2>&1
$PY src/rq1_rq2.py tfidf cnn lstm  > logs/rq1_rq2.log 2>&1
$PY src/rq2_analysis.py            > logs/rq2_analysis.log 2>&1
$PY src/rq3.py tfidf cnn lstm      > logs/rq3.log 2>&1
$PY src/rq4.py 1 2 3               > logs/rq4_part123.log 2>&1
$PY src/rq4.py 4                   > logs/rq4_part4.log 2>&1
$PY src/plots.py                   > logs/plots.log 2>&1
