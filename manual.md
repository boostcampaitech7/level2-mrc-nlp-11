## streamlit dataviewer 사용법

- `chmod +x ./run_streamlit.sh` 명령어로 권한 설정해주고, `./run_streamlit.sh` 해주면 토큰화 결과, prediction 파일 없으면 알아서 생성해준 다음 프로토타입이 실행됩니다. 그냥 프로토타입만 실행하고 싶으면 `streamlit run dataviewer.py`

### 템플릿에서 추가해줘야 하는 부분

- `module/data.py`의 prepare_train_features 함수

  ```python
  # 121번 줄 근처의 아래 코드 아래에
  tokenized_examples["end_positions"] = []
  # 이 코드 추가
  tokenized_examples["example_id"] = []

  # 이후 133줄 근처의 아래 코드 아래에
  answers = examples["answers"][sample_index]
  # 이 코드 추가
  tokenized_examples["example_id"].append(examples["id"][sample_index])
  ```

---

기타 에러 발생하면 알려주세요!
