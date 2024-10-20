import json
import pandas as pd
from datasets import load_from_disk, Dataset, Features, Sequence, Value
import ast
import asyncio
from utils.call_openai_API import call_openai_API
import os
from dotenv import load_dotenv
import glob

# .env 파일 로드
load_dotenv()


# train, valid 데이터를 불러오는 함수
def load_dataset():
    # train, valid 데이터셋 불러오기
    dataset = load_from_disk(
        os.getenv("DIR_PATH") + "level2-mrc-nlp-11/data/train_dataset"
    )

    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]

    train_df = pd.DataFrame(train_dataset)
    valid_df = pd.DataFrame(valid_dataset)

    return train_df, valid_df


# 기존 미활용 문서만 추출하여 저장하는 함수
def filter_unused_document():
    # wikipedia.json 불러오기
    with open(
        os.getenv("DIR_PATH") + "level2-mrc-nlp-11/data/wikipedia_documents.json",
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # wikipedia.json 데이터 프레임 변환
    wikipedia_df = pd.DataFrame(data).transpose()

    # train, valid 데이터셋 불러오기
    train_df, valid_df = load_dataset()

    # train, valid 데이터셋 합치기
    concat_df = pd.concat([train_df, valid_df], ignore_index=True)

    # document_id 기준 중복 데이터 제거하기
    unique_df = concat_df.drop_duplicates(subset="document_id")

    # 관련 질문이 생성되지 않은 문서만 추출하기
    not_in_original = wikipedia_df[
        ~wikipedia_df["document_id"].isin(unique_df["document_id"])
    ]

    # 문서 순서 섞기
    not_in_original = not_in_original.sample(frac=1).reset_index(drop=True)

    # 데이터프레임을 JSON 형식으로 저장
    not_in_original.to_json(
        os.getenv("DIR_PATH")
        + "level2-mrc-nlp-11/data/aug_new/wikipedia_documents.json",
        orient="records",
        force_ascii=False,
    )

    # 분할 크기 설정
    split_size = 5000  # 한 파일에 100개의 항목씩 저장

    for i in range(0, len(not_in_original), split_size):
        # DataFrame을 딕셔너리로 변환 후 일정 크기만큼 슬라이싱
        chunk = not_in_original[i : i + split_size].to_dict(orient="records")

        # 파일 경로 설정
        file_path = (
            os.getenv("DIR_PATH")
            + f"level2-mrc-nlp-11/data/aug_new/chunk_{i // split_size}.json"
        )

        # 파일로 저장
        with open(file_path, "w", encoding="utf-8") as f_out:
            json.dump(chunk, f_out, ensure_ascii=False, indent=4)

    return not_in_original


# OpenAI API를 활용하여 기존 미활용 문서에서 새 질문 답변을 생성하는 함수
def create_new_document_qa():
    # train, valid 데이터셋 불러오기
    train_df, valid_df = load_dataset()

    for i in range(12):
        # 파일 경로 설정
        file_path = (
            os.getenv("DIR_PATH") + f"level2-mrc-nlp-11/data/aug_new/chunk_{i}.json"
        )

        # wikipedia 데이터 불러오기
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        prompt_content = f"""
            다음은 wikipedia의 문서를 정리한 json 파일입니다. 파일의 형식은 다음과 같습니다.
            [
                {{
                    "text":"",
                    "corpus_source":"",
                    "url":"",
                    "domain":null,
                    "title":"",
                    "author":null,
                    "html":null,
                    "document_id":0
                }},
                ...
            ]

            이 파일을 읽고 각 행에 대해 꼭 한 개의 질문과 답변을 생성해주세요. 한 데이터에서 한 가지 초과의 질문과 답변을 생성해서는 안됩니다.
            답변에는 title, context, question, id, answer_start, answer_text, document_id, __index_level_0__가 포함되어야 합니다.
            생성한 질문이 어떤 document_id를 가지는 문서에서 생성된 것인지, 정답은 무엇인지, 정답의 위치는 해당 문서의 몇번째 인덱스에서 시작하는지를 저장해야합니다.
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

        # OpenAI API 새 질문 답변 생성 요청 및 응답 저장하기
        response = asyncio.run(
            call_openai_API(data, 1, prompt_content, response_format)
        )

        # 응답 데이터 프레임으로 변환하기
        response_df = pd.DataFrame(response)

        # csv 파일로 저장하기
        response_df.to_csv("response_df_{i}.csv")

        if i <= 9:  # train에 저장할 질문 답변
            train_df = pd.concat([train_df, train_response_df], ignore_index=True)
        else:  # valid에 저장할 질문 답변
            valid_df = pd.concat([valid_df, valid_response_df], ignore_index=True)

    # train.csv 저장
    train_output_path = (
        os.getenv("DIR_PATH") + "level2-mrc-nlp-11/data/aug_new/train/train.csv"
    )

    train_df.to_csv(train_output_path)

    # valid.csv 저장
    valid_output_path = (
        os.getenv("DIR_PATH") + "level2-mrc-nlp-11/data/aug_new/validation/valid.csv"
    )

    valid_df.to_csv(valid_output_path)

    return response_df


# csv -> arrow 변환 함수
def csv_to_arrow(input_file_path, output_file_path):
    # 데이터 파일 불러오기
    df = pd.read_csv(input_file_path)

    # 불필요한 'Unnamed: 0'과 '__index_level_0__' 컬럼 제거
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    if "__index_level_0__" in df.columns:
        df = df.drop(columns=["__index_level_0__"])

    # answers 데이터에서 answer_start를 int32로 변환
    def convert_answers(answers):
        # 문자열인 경우 딕셔너리로 변환
        if isinstance(answers, str):
            answers = ast.literal_eval(answers)  # 문자열을 딕셔너리로 변환

        # answer_start를 int32로 변환
        answers["answer_start"] = [int(i) for i in answers["answer_start"]]
        return answers

    # answer_start를 int32로 변환 후 저장
    df["answers"] = df["answers"].apply(convert_answers)

    # answer_start를 명시적으로 int32로 변환
    for idx, row in df.iterrows():
        row["answers"]["answer_start"] = [
            int(i) for i in row["answers"]["answer_start"]
        ]

    # features schema 정의
    features = Features(
        {
            "id": Value("string"),
            "title": Value("string"),
            "context": Value("string"),
            "question": Value("string"),
            "answers": {
                "text": Sequence(Value("string")),
                "answer_start": Sequence(
                    Value("int64")
                ),  # 'int64'로 설정하여 데이터 일치
            },
            "document_id": Value("int64"),
        }
    )

    # dataset 생성
    dataset = Dataset.from_pandas(df, features=features)

    # dataset 저장
    dataset.save_to_disk(output_file_path)


# aug_new 폴더 csv 파일, chunk.json 파일 제거 함수
def delete_csv_json_in_aug_new(directory_path):
    # 특정 경로의 모든 CSV 파일을 찾음
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))

    # 각 CSV 파일 삭제
    for file in csv_files:
        os.remove(file)
        print(f"Deleted file: {file}")

    # 특정 경로의 모든 chunk_*.json 파일을 찾음
    json_files = glob.glob(os.path.join(directory_path, "chunk_*.json"))

    # 각 chunk_*.json 파일 삭제
    for file in json_files:
        os.remove(file)
        print(f"Deleted JSON file: {file}")


# aug_new 폴더의 csv파일을 arrow 형식으로 변환, csv 파일 & chunk.json 파일 제거 함수
def convert_and_cleanup():
    # train, valid 파일 경로 설정
    train_input_file_path = (
        os.getenv("DIR_PATH") + "level2-mrc-nlp-11/data/aug_new/train/train_df.csv"
    )
    valid_input_file_path = (
        os.getenv("DIR_PATH") + "level2-mrc-nlp-11/data/aug_new/validation/valid_df.csv"
    )
    train_output_file_path = (
        os.getenv("DIR_PATH") + "level2-mrc-nlp-11/data/aug_new/train"
    )
    valid_output_file_path = (
        os.getenv("DIR_PATH") + "level2-mrc-nlp-11/data/aug_new/validation"
    )

    # train.csv arrow 파일로 변환
    csv_to_arrow(train_input_file_path, train_output_file_path)
    # valid.csv arrow 파일로 변환
    csv_to_arrow(valid_input_file_path, valid_output_file_path)

    # csv 파일, chunk.json 파일 제거
    delete_csv_json_in_aug_new(
        os.getenv("DIR_PATH") + "level2-mrc-nlp-11/data/aug_new/"
    )


if __name__ == "__main__":
    filter_unused_document()
    create_new_document_qa()
    convert_and_cleanup()
