import os
import re
import asyncio
import aiohttp
import pandas as pd
import json
from tqdm import tqdm
from datasets import Dataset

# 환경 변수에 API 키를 저장하는 것을 권장합니다.
# 예: export OPENAI_API_KEY='your-api-key
# 1. OpenAI API 키 설정
OPENAI_API_KEY = ""
# 2. 데이터 로드
documents_path = (
    "/data/ephemeral/home/jaehyeop/level2-mrc-nlp-11/data/wikipedia_documents.json"
)
async_request_number = 10  # 동시 요청 수를 10개로 제한

code_preprocess = [
    "362",
    "1255",
    "3868",
    "4916",
    "5809",
    "7394",
    "8422",
    "9384",
    "11668",
    "15292",
    "34273",
    "34292",
    "41663",
    "45189",
    "49013",
    "50502",
    "50992",
    "54440",
    "59474",
    "59689",
    "59690",
    "13630",
]

with open(documents_path, "r", encoding="utf-8") as f:
    documents = json.load(f)

documents = [(doc_id, documents[doc_id]["text"]) for doc_id in code_preprocess]


# 3. 배치 분할 함수 정의
def split_into_batches(data, batch_size):
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


semaphore = asyncio.Semaphore(async_request_number)


# 4. 비동기 API 호출 함수 정의
async def categorize_batch(session, batch, batch_index):
    async with semaphore:
        docs = "\n".join([rf"{i}: {d}" for i, d in enumerate(batch)])
        prompt = f"""
    다음은 {len(batch)}개의 문서 목록입니다. 이 문서들은 위키피디아 문서의 일부를 크롤링한 문서들이며, 코드 표현 또는 latex 수식 표현이 포함되어 있습니다. 예) 'nunsigned int fib(unsigned int n)', 'main(int argc, char *argv[])', 'integer p = rq(x) + rq(y)'
    각 문서에서 이러한 긴 코드 표현 및 수식 표현을 제거하여 반환해주세요. 단, '할당 : malloc() : 힙영역으로 부터 데이터 공간을 할당 받는다.'에서 'malloc()'과 같이 용어 설명이나 짧은 인라인 코드인 경우는 그대로 유지해주세요.
    응답은 문서 번호와 코드 표현 및 수식 표현을 제거한 문서를 순수한 JSON 형식으로 제공해주세요. 단, 'Invalid \escape' 오류가 발생하지 않도록 텍스트에 이스케이프 문자가 포함된 경우 raw string 처리하여 문자 그대로 취급되게 반환해주세요. 코드 블록이나 다른 텍스트는 포함하지 마세요.

    질문 목록:
    {docs}

    응답 형식:
    [
    {{"document_id": 1, "preprocessed_text": "전처리된 텍스트"}},
    {{"document_id": 3, "preprocessed_text": "코드 표현이 제거된 텍스트입니다."]}},
    ...
    ]
    """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }
        data = {
            "model": "gpt-4o-mini",  # 더 빠른 모델 사용
            "messages": [
                {
                    "role": "system",
                    "content": "You are an assistant that preprocesses texts with removing wikipedia annotation representations.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }

        url = "https://api.openai.com/v1/chat/completions"

        try:
            async with session.post(url, headers=headers, json=data) as response:
                response_json = await response.json()
                response_text = response_json["choices"][0]["message"][
                    "content"
                ].strip()
                response_text = re.sub(r"^```json\s*", "", response_text)
                response_text = re.sub(r"```$", "", response_text)
                preprocessed = json.loads(response_text)
                return batch_index, preprocessed
        except Exception as e:
            print(f"Error in batch {batch_index}: {e}")
            return batch_index, [None] * len(batch)


async def main():
    batch_size = 1
    batches = split_into_batches(documents, batch_size)
    total_batches = len(batches)
    results = [None] * total_batches
    preprocessed_documents = {}

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, batch in enumerate(batches):
            task = asyncio.ensure_future(categorize_batch(session, batch, i))
            tasks.append(task)

        for f in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Processing Batches"
        ):
            batch_index, preprocessed_docs = await f
            results[batch_index] = preprocessed_docs

    for docs in results:
        for doc in docs:
            if not doc:
                continue
            preprocessed_documents[doc["document_id"]] = {
                "text": doc["preprocessed_text"]
            }

    # 6.
    save_path = "/data/ephemeral/home/jaehyeop/level2-mrc-nlp-11/data/documents_removed_code.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(preprocessed_documents, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
