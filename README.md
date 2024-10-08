# Format

## dataset
```JSON
{
    "id": str,
    "title": str,
    "context": str,
    "question": str,
    "answers": {
        "text": List[str],
        "answer_start": List[int]
    }
}
```
## model output
```JSON
# MRC
{
    "predictions": [{
        "id": str,
        "prediction_text": str
    }], //len=(# of data point)
    "label_ids": [{
        "id": str,
        "answers": {
            "text": List[str],
            "answer_start": List[int]
        }
    }] //len=(# of data point)
}

# Retrieval
{
    "sim_score": np.array, //shape=(# of data point, num_neg+1)
    "targets": np.array //shape=(# of data point)
}
```

# configuration
## MRC
- [x] 데이터 전처리: 변경 가능 (modify `pl.datamodule.setup` in `/module/data.py`)
- [ ] ~모델링: 변경 불가 (only use `AutoModelForQuestionAnswering`)~
- [ ] ~손실 함수: 변경 불가 (only use `AutoModelForQuestionAnswering`)~
- [x] 최적화 함수: 변경 가능 (if exists in `torch.optim`)
- [x] 평가 지표: 변경 가능 (if exsits in `datasets_modules.metrics`)

## Retrieval
- [x] 데이터 전처리: 변경 가능 (modify `pl.datamodule.setup` in `/module/data.py`)
- [x] 모델링: 변경 가능 (custom model class in `/module/encoder`)
- [x] 손실 함수: 변경 가능 (custom loss function in `/module/loss`)
- [x] 최적화 함수: 변경 가능 (if exists in `torch.optim`)
- [x] 평가 지표: 변경 가능 (if exsits in `datasets_modules.metrics`)

# TODO
- [x] `wandb` 연결
- [x] `sweep` 기능 추가
- [x] inference 기능 추가
- [ ] checkpoint 및 prediction 파일명 기능 추가
