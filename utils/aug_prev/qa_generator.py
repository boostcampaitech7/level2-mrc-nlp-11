import os
import pandas as pd
import logging
import aiohttp
import asyncio
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry 

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 환경변수에서 API 키 로드
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# 분당 최대 요청 횟수 (예: 60회) 
REQUESTS_PER_MINUTE = 60 

# 요청 간 레이트 리미트 설정 
@sleep_and_retry 
@limits(calls=REQUESTS_PER_MINUTE, period=60) 
async def call_openai_api(prompt_content):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "새로운 질문과 답변 생성을 도와주는 어시스턴트입니다."},
            {"role": "user", "content": prompt_content},
        ],
        "temperature": 0.5,
    }

    async with aiohttp.ClientSession() as session:
        for attempt in range(5):  # 최대 5회 재시도
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data) as response:
                if response.status == 429:  # Too Many Requests
                    logging.warning("API 호출 실패: 429 Too Many Requests. 10초 대기 후 재시도.")
                    await asyncio.sleep(10)  # 대기 후 재시도
                    continue
                elif response.status != 200:
                    logging.error(f"API 호출 실패: {response.status}")
                    return None
                
                response_json = await response.json()
                return response_json["choices"][0]["message"]["content"]

    logging.error("API 호출 실패: 재시도 한도를 초과했습니다.")
    return None


class QAGenerator:
    def __init__(self):
        pass

    async def generate_qa_pairs(self, id, title, context, existing_question, existing_answer):
        # 새로운 질문-답변 쌍을 생성하기 위한 프롬프트 생성
        prompt_content = f"""
            다음 문맥을 기반으로 {existing_question}와는 다른, 질문과 그에 대한 답변을 5개 만들어 주세요.
            문맥: {context}
            기존 질문: {existing_question}
            기존 답변: {existing_answer}
            각 문맥에 대해 질문과 답변은 5개씩 생성해야 하며,
            답변은 문맥에서 직접적으로 나타나는 단어 또는 구문 중 하나에 속해야 하며,
            여러 단어로 이루어진 답변일 경우에도 문맥의 정확한 문장에서 찾아야 합니다. 
            답변은 3자에서 10자 사이여야 하며,
            모든 5개의 응답은 반드시 다음 형식으로 저장되어야 합니다:
            응답 형식: {{
                "id": "{id}",
                "title": "{title}",
                "context": "{context}",
                "question": "<생성된 질문>",
                "answers": 
                {{
                    "answer_start": [<생성된 답변의 시작 인덱스>],
                    "text": ["<생성된 답변>"]
                }}
            }}
        """
        response_content = await call_openai_api(prompt_content)

        # 로그에 생성된 response_content 출력
        logging.info("Generated response content:")
        logging.info(response_content)

        return response_content



    async def augment_data(self, data): # QA 쌍을 만들어서 원래 데이터에 추가시키는 함수
        augmented_data = []
        total_rows = len(data)  # 총 데이터 개수

        for index in range(total_rows):
            row = data.iloc[index]  # 현재 인덱스의 행 데이터 가져오기
            logging.info(f"Train 데이터 인덱스: {index}")  # 현재 인덱스 로그 출력
            id = row['id']
            title = row['title']
            context = row["context"]
            question = row["question"]
            answer = row["answers"]["text"][0:]  # Assuming a single answer

            logging.info(f"Processing context: {context[0:30]} ...(중략)")  # 현재 문맥 로그 출력

            new_data = await self.generate_qa_pairs(id, title, context, question, answer) #생성된 data 출력됨
            if new_data is None:
                logging.warning(f"API 호출 실패로 인해 인덱스 {index}에 대한 데이터가 생성되지 않았습니다.")
                continue  # API 호출 실패 시 다음 질문으로 넘어감
        
            # 로그 진행 상황 표시
            logging.info(f"Processed {index + 1}/{total_rows} contexts.")  # 현재 진행 상황 로그

            augmented_data.append(new_data)  # 생성된 QA 쌍 추가

        return pd.DataFrame(augmented_data)