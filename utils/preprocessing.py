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
from konlpy.tag import Okt, Kkma


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
  

# 전처리 기본 함수
def original(example):
    return example


# 제목을 강조하기 위해 특정 토큰을 추가합니다.
def title_context_merge_token(example):
    title = f"<TITLE> {example['title']} <TITLE_END> "
    
    # 각 답변의 시작 위치를 수정
    for idx in range(len(example["answers"]["answer_start"])):
        example["answers"]["answer_start"][idx] = example["answers"]["answer_start"][
            idx
        ] + len(title)
    
    # 제목과 문맥을 결합
    example["context"] = f"{title}{example['context']}"
    return example


# 제목을 컨텍스트 앞에 추가하고, 각 답변의 시작 위치를 제목 길이만큼 조정하는 함수.
def title_context_merge_gap(example):
    title = example['title']

    # 각 답변의 시작 위치를 수정
    for idx in range(len(example["answers"]["answer_start"])):
        example["answers"]["answer_start"][idx] = example["answers"]["answer_start"][
            idx
        ] + len(title) + 1
    
    # 제목과 문맥을 공백으로 결합
    example["context"] = f"{title} {example['context']}"
    return example


# 제목과 질문 간의 Jaccard 유사도를 기반으로 제목을 강조하여 컨텍스트에 포함하는 함수.
def title_context_merge_jaccard(example):
    # 제목과 문서 전처리
    title = example['title'].strip()
    context = example['context'].strip()
    question = example['question'].strip()

    # 제목과 질문의 단어 집합 생성
    title_words = set(re.findall(r'\w+', title))
    question_words = set(re.findall(r'\w+', question))

    # Jaccard 유사도 계산
    jaccard_similarity = len(title_words.intersection(question_words)) / len(title_words.union(question_words))

    # 제목이 질문과 관련이 높을 경우 컨텍스트에 제목을 강조
    if jaccard_similarity > 0.7:  # 유사성이 일정 기준 이상일 때
        example['context'] = f"<TITLE> {title} <TITLE_END> {context}"  # 스페셜 토큰 추가
        for idx in range(len(example["answers"]["answer_start"])):
            example["answers"]["answer_start"][idx] = example["answers"]["answer_start"][idx] + len(example['title'])
    return example


# 제목을 컨텍스트 뒤에 강조된 형태로 추가하는 함수.
def title_context_merge_behind(example):
    # 제목을 강조하기 위해 특정 토큰을 추가합니다.
    title = f" <TITLE> {example['title']} <TITLE_END>"

    example["context"] = f"{example['context']}{title}"
    return example

 
# question 조사 제거 함수
def remove_josa(example):
    okt = Okt()
    tokens = okt.pos(example['question'])
    filtered_tokens = [word for word, pos in tokens if pos != 'Josa']
    example['question'] = example['question'] + " [SEP] " + ' '.join(filtered_tokens)
    print(example['question'])
    return example

  
# question 명사 추출 및 병합 함수 
def nouns(example):
    kkma = Kkma()
    tokens = kkma.pos(example['question'])

    filtered_tokens = [word for word, pos in tokens if pos in ['NNG', 'NNP', 'NNB', 'NP', 'NR']]

    # 복합 명사 처리
    merged_tokens = []
    temp = ""
    for word in filtered_tokens:
        if temp:  # 이전에 처리된 명사가 있으면 결합 시도
            # 예시: '행' + '정부' => '행정부'
            merged_word = temp + word
            if merged_word in example['question']:  # 결합된 단어가 원본 텍스트에 있으면 결합
                temp = merged_word  # 결합한 단어를 temp에 저장
            else:  # 결합 실패하면 temp에 현재 단어 저장
                merged_tokens.append(temp)
                temp = word
        else:
            temp = word

    if temp:  # 마지막 남은 단어 처리
        merged_tokens.append(temp)

    example['question'] = example['question'] + " [SEP] " + ', '.join(merged_tokens)

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
