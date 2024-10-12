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

import re
from transformers import BertTokenizerFast


def test(example):
    example["context"] = example["context"].replace("\n", " ")
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


# 답변 시작 토큰과 같은 토큰값 중 답변 시작 토큰의 인덱스 저장 함수
def find_target_index(tokens, target_token):
    occurrence_index = 0
    for i in range(len(tokens)):
        if tokens[i] == target_token:
            occurrence_index += 1
    return occurrence_index - 1


# 마크다운 패턴 제거 함수
def remove_markdown(example):
    # 원본 문맥과 answer_start 저장
    original_context = example["context"]
    original_answer_start = example["answers"]["answer_start"][0]

    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")

    # 원본 문맥을 토큰화
    original_tokens = tokenizer(original_context)["input_ids"]

    # 답변 시작 텍스트 토큰화 결과 저장
    answer_token = original_tokens[original_answer_start]

    # 답변 시작 토큰이 같은 토큰값 중 몇 번째 인덱스에 위치하는지 저장
    original_index = find_target_index(original_tokens, answer_token)

    # 마크다운 패턴 리스트를 함수 내부에 포함
    markdown_patterns = {
        "header": r"(?:^|\n{1,2})#{1,6}\s*",  # 헤더 (1~2번의 줄바꿈 허용)
        "list": r"(?:\\n|\\n\\n|\n|\n\n)\s*[*+-]{1,}\s+",  # 리스트 기호 탐지 (\\n, \n, 공백 허용)
    }

    # 마크다운 문법에 해당하는 부분을 제거
    for pattern_name, pattern in markdown_patterns.items():
        example["context"] = re.sub(
            pattern, " ", example["context"], flags=re.MULTILINE
        )

    # 마크다운을 제거한 후 새로운 문맥을 토큰화
    new_tokens = tokenizer(example["context"])["input_ids"]

    # 답변 시작 토큰이 같은 토큰값의 인덱스들을 찾음
    new_indices = [i for i, token in enumerate(new_tokens) if token == answer_token]

    # original_index번째 값을 example['answers']['answer_start']에 저장
    if original_index < len(new_indices):
        example["answers"]["answer_start"] = new_indices[original_index]
    else:
        example["answers"]["answer_start"] = None  # 또는 예외 처리

    return example


# 마크다운 패턴별 스페셜 토큰 추가 함수
def replace_markdown_with_tags(example):
    # 원본 문맥과 answer_start 저장
    original_context = example["context"]
    original_answer_start = example["answers"]["answer_start"][0]

    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")

    # 원본 문맥을 토큰화
    original_tokens = tokenizer(original_context)["input_ids"]

    # 답변 시작 텍스트 토큰화 결과 저장
    answer_token = original_tokens[original_answer_start]

    # 답변 시작 토큰이 같은 토큰값 중 몇 번째 인덱스에 위치하는지 저장
    original_index = find_target_index(original_tokens, answer_token)

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
        example["context"] = re.sub(
            pattern, replacement, example["context"], flags=re.MULTILINE
        )  # 패턴에 해당하는 부분을 태그로 대체

    # 마크다운을 제거한 후 새로운 문맥을 토큰화
    new_tokens = tokenizer(example["context"])["input_ids"]

    # 답변 시작 토큰이 같은 토큰값의 인덱스들을 찾음
    new_indices = [i for i, token in enumerate(new_tokens) if token == answer_token]

    # original_index번째 값을 example['answers']['answer_start']에 저장
    if original_index < len(new_indices):
        example["answers"]["answer_start"] = new_indices[original_index]
    else:
        example["answers"]["answer_start"] = None  # 또는 예외 처리

    return example


# 마크다운 패턴 <DOC> 스페셜 토큰 추가 함수
def replace_markdown_with_doc(example):
    # 원본 문맥과 answer_start 저장
    original_context = example["context"]
    original_answer_start = example["answers"]["answer_start"][0]

    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")

    # 원본 문맥을 토큰화
    original_tokens = tokenizer(original_context)["input_ids"]

    # 답변 시작 텍스트 토큰화 결과 저장
    answer_token = original_tokens[original_answer_start]

    # 답변 시작 토큰이 같은 토큰값 중 몇 번째 인덱스에 위치하는지 저장
    original_index = find_target_index(original_tokens, answer_token)

    # 마크다운 패턴 리스트를 함수 내부에 포함
    markdown_patterns = {
        "header": r"(?:^|\n{1,2})#{1,6}\s*",  # 헤더 (ex: #, ##, ###, ...)
        "list": r"(?:\\n|\\n\\n|\n|\n\n)\s*[*+-]{1,}\s+",  # 리스트 (ex: *, -, +)
    }

    # 각 마크다운 문법에 해당하는 부분을 <DOC>로 대체하는 과정
    for pattern_name, pattern in markdown_patterns.items():
        example["context"] = re.sub(
            pattern, r"<DOC>", example["context"], flags=re.MULTILINE
        )

    # 마크다운을 제거한 후 새로운 문맥을 토큰화
    new_tokens = tokenizer(example["context"])["input_ids"]

    # 답변 시작 토큰이 같은 토큰값의 인덱스들을 찾음
    new_indices = [i for i, token in enumerate(new_tokens) if token == answer_token]

    # original_index번째 값을 example['answers']['answer_start']에 저장
    if original_index < len(new_indices):
        example["answers"]["answer_start"] = new_indices[original_index]
    else:
        example["answers"]["answer_start"] = None  # 또는 예외 처리

    return example
