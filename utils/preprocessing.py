"""
STANDARD DATA FORMAT
{
    'id': Value(dtype='string', id=None),
    'title': Value(dtype='string', id=None),
    'context': Value(dtype='string', id=None),
    'question': Value(dtype='string', id=None),
    'answers': Sequence(feature={
        'text': Value(dtype='string', id=None),
        'answer_start': Value(dtype='int32', id=None)
    }, length=-1, id=None)
}
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import threading
import re


# 질문의 카테고리를 분류하는 모델
class ModelPredictorSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ModelPredictorSingleton, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path, tokenizer_path, label_encoder_path, device=None):
        if self._initialized:
            return
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.label_encoder_path = label_encoder_path
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self._initialized = True

    def load_resources(self):
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            )
            self.model.to(self.device)
            self.model.eval()
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if self.label_encoder is None:
            self.label_encoder = joblib.load(self.label_encoder_path)

    def predict(self, text):
        self.load_resources()
        # 예측 로직은 이전과 동일합니다.
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).cpu().item()
            category = self.label_encoder.inverse_transform([predicted_class])[0]
        return category


# 질문을 카테고리로 분류 후 오른쪽에 더해줌
def categorize_question(example):
    # Singleton 인스턴스 생성 (필요할 때 한 번만 생성)
    model_predictor = ModelPredictorSingleton(
        model_path="/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/question_classification/7_category_bert-base-multilingual-cased",
        tokenizer_path="/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/question_classification/7_category_bert-base-multilingual-cased",
        label_encoder_path="/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/question_classification/7_category_bert-base-multilingual-cased/label_encoder.joblib",
    )
    category = model_predictor.predict(example["question"])
    example["question"] = example["question"] + "<" + category + ">"
    return example


def original(example):
    return example


def title_context_merge(example):
    # 제목을 강조하기 위해 특정 토큰을 추가합니다.
    title = f"<TITLE> {example['title']} <TITLE_END> "
    for idx in range(len(example["answers"]["answer_start"])):
        example["answers"]["answer_start"][idx] = example["answers"]["answer_start"][
            idx
        ] + len(title)
    example["context"] = f"{title}{example['context']}"
    return example


# 마크다운 패턴 제거 함수
def remove_markdown(example):
    # 원본 문맥과 answer_start 저장
    original_context = example["context"]
    original_answer_start = example["answers"]["answer_start"][0]

    # 마크다운 패턴 리스트를 함수 내부에 포함
    markdown_patterns = {
        "header": r"(?:^|\n{1,2})#{1,6}\s*",  # 헤더 (1~2번의 줄바꿈 허용)
        "list": r"(?:\\n|\\n\\n|\n|\n\n)\s*[*+-]{1,}\s+",  # 리스트 기호 탐지 (\\n, \n, 공백 허용)
    }

    # 마크다운 문법에 해당하는 부분을 제거 및 answer_start 조정
    for pattern_name, pattern in markdown_patterns.items():
        match_iter = re.finditer(pattern, original_context, flags=re.MULTILINE)
        for match in match_iter:
            if match.start() < original_answer_start:
                # 마크다운 패턴이 answer_start보다 앞에 있으면 그 길이만큼 answer_start를 줄임
                original_answer_start -= len(match.group(0))

        example["context"] = re.sub(
            pattern, " ", example["context"], flags=re.MULTILINE
        )

    # 변경된 answer_start 값을 적용
    example["answers"]["answer_start"] = [original_answer_start]

    return example


# 마크다운 패턴별 스페셜 토큰 추가 함수
def replace_markdown_with_tags(example):
    # 원본 문맥과 answer_start 저장
    original_context = example["context"]
    original_answer_start = example["answers"]["answer_start"][0]

    # 마크다운 패턴 리스트를 함수 내부에 포함
    markdown_patterns = {
        "header": (
            r"(?:^|\n{1,2})#{1,6}\s*",
            r"<HEADER>",
        ),  # 헤더 (ex: #, ##, ###, ...) - <header>로 대체
        "list": (
            r"(?:\\n|\\n\\n|\n|\n\n)\s*[*+-]{1,}\s+",
            r"<LIST>",
        ),  # 리스트 (ex: *, -, +) - 리스트 기호 제거하고 <list>로 대체
    }

    # 각 마크다운 문법에 해당하는 부분을 태그로 대체하는 과정
    for pattern, replacement in markdown_patterns.values():
        match_iter = re.finditer(pattern, original_context, flags=re.MULTILINE)
        for match in match_iter:
            if match.start() < original_answer_start:
                # 마크다운 패턴이 answer_start보다 앞에 있으면 그 길이만큼 answer_start를 늘림 (스페셜 토큰의 길이만큼)
                original_answer_start += len(replacement) - len(match.group(0))

        example["context"] = re.sub(
            pattern, replacement, example["context"], flags=re.MULTILINE
        )  # 패턴에 해당하는 부분을 태그로 대체

    # 변경된 answer_start 값을 적용
    example["answers"]["answer_start"] = [original_answer_start]

    return example


# 마크다운 패턴 <DOC> 스페셜 토큰 추가 함수
def replace_markdown_with_doc(example):
    # 원본 문맥과 answer_start 저장
    original_context = example["context"]
    original_answer_start = example["answers"]["answer_start"][0]

    # 마크다운 패턴 리스트를 함수 내부에 포함
    markdown_patterns = {
        "header": r"(?:^|\n{1,2})#{1,6}\s*",  # 헤더 (ex: #, ##, ###, ...)
        "list": r"(?:\\n|\\n\\n|\n|\n\n)\s*[*+-]{1,}\s+",  # 리스트 (ex: *, -, +)
    }

    # 각 마크다운 문법에 해당하는 부분을 <DOC>로 대체하는 과정
    for pattern_name, pattern in markdown_patterns.items():
        match_iter = re.finditer(pattern, original_context, flags=re.MULTILINE)
        for match in match_iter:
            if match.start() < original_answer_start:
                # 마크다운 패턴이 answer_start보다 앞에 있으면 그 길이만큼 answer_start를 늘림 (스페셜 토큰의 길이만큼)
                original_answer_start += len("<DOC>") - len(match.group(0))

        example["context"] = re.sub(
            pattern, r"<DOC>", example["context"], flags=re.MULTILINE
        )

    return example
