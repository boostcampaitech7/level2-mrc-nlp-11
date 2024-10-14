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
    # example["context"] = example["context"].replace("\n", " ")
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
