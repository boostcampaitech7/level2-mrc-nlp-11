import asyncio
import aiohttp
import json
import re
from tqdm import tqdm
from dotenv import load_dotenv
import os


async def call_openai_API(data, batch_size, prompt_content, response_format):
    load_dotenv()
    # 1. OpenAI API 키 설정
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    async_request_number = 10

    # 3. 배치 분할 함수
    def split_into_batches(data, batch_size):
        return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    semaphore = asyncio.Semaphore(async_request_number)

    # 4. 비동기 API 호출 함수
    async def process_batch(
        session, batch, batch_index, prompt_content, response_format
    ):
        async with semaphore:
            docs = "\n".join([rf"{i}: {d}" for i, d in enumerate(batch)])
            prompt = f"""
                {prompt_content}

                문서 목록:
                {docs}

                응답 형식:
                {response_format}
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
                        "content": "You are an assistant that preprocesses texts following requirements.",
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

    batches = split_into_batches(data, batch_size)
    total_batches = len(batches)
    results = [None] * total_batches
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, batch in enumerate(batches):
            task = asyncio.ensure_future(
                process_batch(session, batch, i, prompt_content, response_format)
            )
            tasks.append(task)

        for f in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Processing Batches"
        ):
            batch_index, preprocessed_docs = await f
            results[batch_index] = preprocessed_docs

    return results
