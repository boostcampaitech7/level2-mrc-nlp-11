import json
import pandas as pd
from datasets import load_from_disk
import asyncio
from call_openai_API import call_openai_API
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


# 데이터 불러오기
def load_dataset():
    # train, valid 데이터셋 불러오기
    dataset = load_from_disk(
        f"file://{os.getenv('DIR_PATH')}level2-mrc-nlp-11/data/train_dataset"
    )

    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]

    train_df = pd.DataFrame(train_dataset)
    valid_df = pd.DataFrame(valid_dataset)

    # 데이터 확인 로그 추가
    # print("Train dataset sample:")
    # print(train_df.iloc[[0]])  # 첫 행 출력
    # print("Validation dataset sample:")
    # print(valid_df.iloc[[0]])  # 첫 행 출력

    return train_df, valid_df


# 기존 활용 문서 중 중복되지 않은 것만 추출하여 저장하는 함수
def filter_used_document_prev():

    train_df, valid_df = load_dataset()

    duplicates = pd.merge(train_df, valid_df, on="document_id")

    # train_df, valid_df에서 중복된 문서 제거
    train_df = train_df[~train_df["document_id"].isin(duplicates["document_id"])]
    valid_df = valid_df[~valid_df["document_id"].isin(duplicates["document_id"])]

    # context에서 \n 문자 제거 (경우에 따라)
    # train_df['context'] = train_df['context'].str.replace('\\n', ' ', regex=False)
    # valid_df['context'] = valid_df['context'].str.replace('\\n', ' ', regex=False)

    print(train_df.head()['context'])

    return train_df, valid_df


def create_prev_document_qa():

    train_df, valid_df = filter_used_document_prev()

    prompt_content = f"""
        Dataframe Structure:
        [
            {{
                "title": "",
                "context": "",
                "question": "",
                "id": "",
                "answers":
                {{
                        'answer_start': [],
                        'text': []
                }},
                "document_id": "",
                "__index_level_0__": ""
            }},
            ...
        ]

        Request: Generate five question and answer for the text of each article list. 
        Generate only one question and answer in one piece of data. 
        The format of the answer should follow the format of the response. 
        Please ensure that whenever quotation marks (") are included in the original or modified sentence, 
        escape characters (\") are added to prevent errors during the JSON parsing process.
        The answer must be found inside the context. 
        
        The definition of answer_start is the character index where the answer starts in the context, counting from the beginning of the context, including spaces and punctuation. 
        For example, if the context is 'Hi, my name is Tony, ... ', and the answer is 'Tony', then the value of answer_start is 15, because the index starts counting from the very first character.
        """

    response_format = f"""
        [
            {{
                "title": "",
                "context": "",
                "question": "",
                "id": "",
                "answers":
                {{
                        'answer_start': [],
                        'text': []
                }},
                "document_id": "",
                "__index_level_0__": ""
            }},
            ...
        ]
        """

    train_data = train_df.to_dict(
        orient="records"
    )  # DataFrame을 JSON 형식의 리스트로 변환
    valid_data = valid_df.to_dict(orient="records")

    response_1 = asyncio.run(  # train(중복된 거 뺀)
        call_openai_API(train_data, 1, prompt_content, response_format)
    )

    for i in range(len(response_1)):
        response_1.at[i, 'id'] = f"{response_1.at[i, 'id']}_{i}"

    response_2 = asyncio.run(  # valid(중복된 거 뺀)
        call_openai_API(valid_data, 1, prompt_content, response_format)
    )

    for i in range(len(response_2)):
        response_2.at[i, 'id'] = f"{response_1.at[i, 'id']}_{i}"

    train_df = pd.concat([train_df, response_1], ignore_index=True)  # train 증강

    valid_df = pd.concat([valid_df, response_2], ignore_index=True)  # valid 증강

    # save
    output_path_1 = (  # train 저장 경로
        os.getenv("DIR_PATH")
        + "level2-mrc-nlp-11/data/aug_prev/train/aug_prev_train_data.csv"
    )

    output_path_2 = (  # valid 저장 경로
        os.getenv("DIR_PATH")
        + "level2-mrc-nlp-11/data/aug_prev/train/aug_prev_valid_data.csv"
    )

    train_df.to_csv(output_path_1, index=False)
    valid_df.to_csv(output_path_2, index=False)


if __name__ == "__main__":
    create_prev_document_qa()
