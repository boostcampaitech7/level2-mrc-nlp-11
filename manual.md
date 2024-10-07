### streamlit dataviewer 사용법

템플릿과 제대로 합쳐져 있지 않은 현재 버전에서 사용법입니다. (템플릿 완성 후 템플릿에 결합되도록 수정/보완 예정)
streamlit prototype 실행은 터미널에서 `streamlit run dataviewer.py` 입력하시면 됩니다. (파일 경로 확인)

1. 베이스라인 코드로 사용하는 경우

- utils_qa.py의 postprocess_qa_predictions 함수에서 `pred["start"] = offsets[0]` 부분과 `predioctions.insert({... "start": 0})` 부분을 추가해줍니다. nbest_prediction json 파일에 예측 시작 위치를 포함해주기 위함입니다.

  ```python
  # utils_qa.py
  def postprocess_qa_predictions():
    ...
    for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]
            pred["start"] = offsets[0]  # 추가한 부분

        # rare edge case에는 null이 아닌 예측이 하나도 없으며 failure를 피하기 위해 fake prediction을 만듭니다.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):

            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0, "start": 0}
            )
  ```

- train.py의 run_mrc 함수에서 222줄 쯤에 있는 `train_datset = train_dataset.map()` 과 274줄 쯤에 있는 `eval_dataset = eval_dataset.map()` 아래에 각각 토큰화 결과 저장 코드 삽입

  ```python
  # dataset에서 train feature를 생성합니다.
  train_dataset = train_dataset.map(
      prepare_train_features,
      batched=True,
      num_proc=data_args.preprocessing_num_workers,
      remove_columns=column_names,
      load_from_cache_file=not data_args.overwrite_cache,
  )

  print("===Train Dataset Tokenized=====")
  train_tokenized_samples = {
      train_dataset[i]['example_id'] : tokenizer.convert_ids_to_tokens(train_dataset['input_ids']) for i in range(len(train_dataset))
  }
  file_path = os.path.join(
          "/data/ephemeral/home/jaehyeop/code/outputs/train_dataset/",
          "train_tokenized_samples.json"
      )
  with open(file_path, "w", encoding="utf-8") as writer:
      writer.write(
          json.dumps(train_tokenized_samples, indent=4, ensure_ascii=False) + "\n"
      )
  print("================================")
  ```

  ```python
  # Validation Feature 생성
  eval_dataset = eval_dataset.map(
      prepare_validation_features,
      batched=True,
      num_proc=data_args.preprocessing_num_workers,
      remove_columns=column_names,
      load_from_cache_file=not data_args.overwrite_cache,
  )

  print("====Eval Dataset Tokenized====")
  eval_tokenized_samples = {
      eval_dataset[i]['example_id'] : tokenizer.convert_ids_to_tokens(eval_dataset[i]['input_ids']) for i in range(len(eval_dataset))
  }
  file_path2 = os.path.join(
          "/data/ephemeral/home/jaehyeop/code/outputs/train_dataset/",
          "eval_tokenized_samples.json"
      )
  with open(file_path2, "w", encoding="utf-8") as writer:
      writer.write(
          json.dumps(eval_tokenized_samples, indent=4, ensure_ascii=False) + "\n"
      )
  print("===============================")
  ```

---

2. 템플릿을 사용하는 경우

- save_tokenized_samples.py에서 MRCDataModule 불러오는 경로 잘 설정됐는지 확인
- config.yaml에서

  ```yaml
  tokenizer:
    save: True
  ```

  이렇게 설정 추가해주세요.

---

기타 에러 발생하면 알려주세요!
