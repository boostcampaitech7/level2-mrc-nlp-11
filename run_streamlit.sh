#!/bin/bash

# utils/save_tokenized_samples.py 실행
python3 utils/save_tokenized_samples.py

# eval_nbest_predictions.json이 없으면 train_mrc.py 실행
if [ ! -f "./outputs/eval_nbest_predictions.json" ]; then
  echo "eval_nbest_predictions.json이 없습니다. train_mrc.py를 실행합니다."
  ./run_mrc.sh
fi

# dataviewer.py 실행
streamlit run dataviewer.py
