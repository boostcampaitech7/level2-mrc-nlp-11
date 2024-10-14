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

from konlpy.tag import Okt, Kkma

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


# 조사를 제거하는 함수
def remove_josa(example):
    okt = Okt()
    tokens = okt.pos(example['question'])
    filtered_tokens = [word for word, pos in tokens if pos != 'Josa']
    example['question'] = example['question'] + " [SEP] " + ' '.join(filtered_tokens)
    print(example['question'])
    return example

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
    print(example['question'])
    return example