import os
import re
from openai import OpenAI
import pandas as pd
import math
import time
from tqdm import tqdm
from typing import List
from datasets import Dataset
import json


# 1. OpenAI API 키 설정
client = OpenAI(api_key="")
# 2. 데이터 로드
wiki_path = (
    "/data/ephemeral/home/jaehyeop/level2-mrc-nlp-11/data/wikipedia_documents.json"
)
if os.path.exists(wiki_path):
    with open(wiki_path, "r", encoding="utf-8") as f:
        wiki_documents = json.load(f)

wiki_texts = [wiki_document["text"] for wiki_document in wiki_documents.values()]


def find_taglike_words(text):
    # <...> 형태의 패턴
    pattern = r"<(?![^<>]*[가-힣])[^<>]+>"

    # 패턴에 맞는 모든 단어 찾기
    matches = re.findall(pattern, text)

    return matches


taglike_words = []
for text in wiki_texts:
    words = find_taglike_words(text)
    if words:
        taglike_words.extend(words)
taglike_words = list(set(taglike_words))


# 3. 배치 분할 함수 정의
def split_into_batches(data: List[str], batch_size: int) -> List[List[str]]:
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


# 4. ChatGPT API 호출 함수 정의
def categorize_texts(batch: List[str]) -> List[str]:
    """
    주어진 질문 배치를 카테고리로 분류하고 리스트로 반환합니다.
    """
    # 프롬프트 작성
    prompt = f"""
            다음은 100개의 태그 형식 문자열(<> 사이에 문자열이 있는 형태)입니다. 각 문자열을 다음 세 가지 카테고리 중 하나로 분류해주세요: HTML_TAG, CODE_REPRESENTATION, PROPER_NOUN.
            응답은 문자열을 key, 카테고리를 value로 하는 순수한 JSON 형식으로 제공해주세요. 코드 블록이나 다른 텍스트는 포함하지 마세요.

            질문 목록:
            {batch}

            응답 형식:
            [
                {{"string1": "HTML_TAG"}},
                {{"string2": "PROPER_NOUN"}},
                ...
            ]
            """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 사용하려는 모델 이름
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that categorizes strings into predefined categories.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,  # 응답의 창의성을 낮추기 위해 0으로 설정
        )
        response_text = response.choices[0].message.content.strip()
        response_text = re.sub(r"^```json\s*", "", response_text)
        response_text = re.sub(r"```$", "", response_text)

        # JSON 파싱
        import json

        categorized = json.loads(response_text)

        return categorized

    except Exception as e:
        print(f"Error during API call: {e}")
        # 실패 시 빈 리스트 반환
        return [None] * len(batch)


# 5. 배치 단위로 카테고리 분류
batch_size = 100
batches = split_into_batches(taglike_words, batch_size)
total_batches = len(batches)
categorized_results = {}

print(f"총 문자열 수: {len(taglike_words)}")
print(f"총 배치 수: {total_batches}")

for i, batch in enumerate(tqdm(batches, desc="Processing Batches")):
    categories = categorize_texts(batch)
    print(categories)
    for item in categories:
        for k, v in item.items():
            categorized_results[k] = v

    # API 호출 간 간격을 두어 rate limit 초과 방지
    time.sleep(1)  # 필요에 따라 조정


# 7. 결과 저장
output_path = "/data/ephemeral/home/jaehyeop/level2-mrc-nlp-11/data/default/categorized_taglike_words_2.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(categorized_results, f, ensure_ascii=False, indent=4)
print(f"Categorized data saved to {output_path}")
