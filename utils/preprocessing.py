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


def test(example):
    example["context"] = example["context"].replace("\n", " ")
    return example


def title_context_merge(example):
    title = example['title']

    # 각 답변의 시작 위치를 수정
    for idx in range(len(example["answers"]["answer_start"])):
        example["answers"]["answer_start"][idx] = example["answers"]["answer_start"][
            idx
        ] + len(title) + 1
    
    # 제목과 문맥을 공백으로 결합
    example["context"] = f"{title} {example['context']}"
    return example
