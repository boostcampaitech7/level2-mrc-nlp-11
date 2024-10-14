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


def test(example):
    example["context"] = example["context"].replace("\n", " ")
    return example


def title_context_merge_token(example):
    # 제목을 강조하기 위해 특정 토큰을 추가합니다.
    title = f"<TITLE> {example['title']} <TITLE_END> "
    
    # 각 답변의 시작 위치를 수정
    for idx in range(len(example["answers"]["answer_start"])):
        example["answers"]["answer_start"][idx] = example["answers"]["answer_start"][
            idx
        ] + len(title)
    
    # 제목과 문맥을 결합
    example["context"] = f"{title}{example['context']}"
    return example


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


def title_context_merge_behind(example):
    # 제목을 강조하기 위해 특정 토큰을 추가합니다.
    title = f" <TITLE> {example['title']} <TITLE_END>"

    example["context"] = f"{example['context']}{title}"
    return example