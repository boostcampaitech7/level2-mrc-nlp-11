import json
import pandas as pd
from datasets import load_from_disk, Dataset, Features, Sequence, Value
import ast
import asyncio
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from call_openai_API import call_openai_API
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

    # 데이터프레임을 JSON 형식으로 저장
    not_in_original.to_json(
        os.getenv("DIR_PATH")
        + "level2-mrc-nlp-11/data/aug_new/wikipedia_documents.json",
        orient="records",
        force_ascii=False,
    )

    # 분할 크기 설정
    split_size = 1000  # 한 파일에 1000개의 항목씩 저장

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
    for i in range(50):
        # 파일 경로 설정
        file_path = (
            os.getenv("DIR_PATH") + f"level2-mrc-nlp-11/data/aug_new/chunk_{i}.json"
        )

        # wikipedia 데이터 불러오기
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        prompt_content = f"""
            1. Data formats
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

            Request: Generate one question and a single-word answer for the text of each article in the list.
            Ensure that the question is formulated so that the answer is directly taken from the context of the article and consists of only one or two words.
            Generate only one question and answer pair per data item.
            The format of the answer should follow the format of the response.
            Please ensure that whenever quotation marks (") are included in the original or modified sentence,
            escape characters (\") are added to prevent errors during the JSON parsing process.
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
                        'answer_start': [0],
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
        response_df.to_csv(
            os.getenv("DIR_PATH")
            + f"level2-mrc-nlp-11/data/aug_new/response_df_{i}.csv"
        )

    return response_df


# OpenAI API 응답으로 생성된 csv 파일 정제 함수
def process_and_combine_csv(directory_path, output_path):
    # csv 파일을 불러오기
    all_csv_files = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.endswith(".csv")
    ]

    # 각 csv 파일을 pandas DataFrame으로 불러와 리스트에 저장
    df_list = []

    for file in all_csv_files:
        try:
            df = pd.read_csv(file)

            # 'Unnamed: 0' 컬럼이 있을 경우 제거
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])

            # 모든 값이 NaN인 컬럼 제거
            df = df.dropna(axis=1, how="all")

            # 모든 값이 NaN인 빈 행을 제거
            df = df.dropna(how="all")

            df_list.append(df)
        except Exception as e:
            print(f"파일을 읽는 중 오류 발생: {file}, 오류: {e}")

    # 모든 데이터프레임을 하나로 병합
    combined_df = pd.concat(df_list, ignore_index=True)

    # 첫 번째 컬럼에서 JSON 데이터를 파싱
    parsed_data = []

    for index, row in combined_df.iterrows():
        try:
            # 문자열을 딕셔너리로 변환
            parsed_row = ast.literal_eval(row[0])
            parsed_data.append(parsed_row)
        except Exception as e:
            print(f"Error parsing row {index}: {e}")

    # 변환된 데이터를 데이터프레임으로 변환
    parsed_df = pd.json_normalize(parsed_data)

    # 'answers' 컬럼을 원하는 형식으로 변환하는 함수
    def format_answers(row):
        return {
            "answer_start": (
                row["answers.answer_start"]
                if isinstance(row["answers.answer_start"], list)
                else [row["answers.answer_start"]]
            ),
            "text": (
                row["answers.text"]
                if isinstance(row["answers.text"], list)
                else [row["answers.text"]]
            ),
        }

    # 'answers' 컬럼을 형식에 맞게 변환
    parsed_df["answers"] = parsed_df.apply(format_answers, axis=1)

    # 변환이 완료된 후 'answers.answer_start'와 'answers.text' 컬럼 제거
    parsed_df = parsed_df.drop(columns=["answers.answer_start", "answers.text"])

    # 컬럼 순서를 지정된 순서로 변경
    parsed_df = parsed_df[
        [
            "title",
            "context",
            "question",
            "id",
            "answers",
            "document_id",
            "__index_level_0__",
        ]
    ]

    # document_id를 int 형식으로 변환
    parsed_df["document_id"] = parsed_df["document_id"].astype(int)

    # id 값을 'mrc-{i}' 형식으로 변경
    parsed_df["id"] = ["mrc-{}".format(i) for i in range(len(parsed_df))]

    # document_id 기준으로 정렬
    parsed_df = parsed_df.sort_values(by="document_id")

    # 정제된 데이터프레임 저장
    parsed_df.to_csv(output_path, index=False)

    return parsed_df


# answer_start 계산 함수
def count_escape_chars(text):
    """
    JSON에서 이스케이프가 필요한 문자의 개수를 반환합니다.

    이스케이프가 필요한 문자:
    - 큰따옴표 (")
    - 역슬래시 (\)
    - 제어 문자 (U+0000 ~ U+001F)

    Args:
        text (str): 검사할 문자열

    Returns:
        int: 이스케이프가 필요한 문자의 수
    """
    # 이스케이프가 필요한 문자를 찾는 정규표현식
    # 큰따옴표, 역슬래시, 제어 문자들
    pattern = r'["\\\b\f\n\r\t]|[\x00-\x1F]'

    # 모든 일치하는 문자 검색
    tokens = re.findall(pattern, text)

    # 일치하는 문자의 개수 반환
    return len(tokens)


# answer_start 계산 함수
def update_answer_start(data, directory_path):
    filtered_data = []
    for idx, item in data.iterrows():
        context = item.get("context", "")

        # 'answers'는 이미 딕셔너리 형태라고 가정하고 그대로 사용
        answers = item.get("answers", {"answer_start": [], "text": []})

        answer_texts = answers.get("text", [])
        answer_starts = answers.get("answer_start", [])

        # 모든 answer_text가 context 안에 존재하는지 판단
        all_present = True
        new_answer_starts = []

        for answer_text in answer_texts:
            start_idx = context.find(answer_text)
            if start_idx == -1:
                all_present = False
                break
            # escape 문자 적용
            start_idx += count_escape_chars(context[:start_idx])
            new_answer_starts.append(start_idx)

        if all_present:
            # answer_start 수정
            item["answers"] = {"answer_start": new_answer_starts, "text": answer_texts}
            filtered_data.append(item)

    add_answer_start_df = pd.DataFrame(filtered_data)

    # 결과를 CSV 파일로 저장
    add_answer_start_df.to_csv(directory_path + "add_answer_start.csv", index=False)

    return add_answer_start_df


# train, valid 데이터에 새로운 데이터 병합
def combine_train_valid(data, directory_path):
    # train, valid 데이터셋 불러오기
    train_df, valid_df = load_dataset()

    train_df = pd.concat([train_df, data[:-120]])
    valid_df = pd.concat([valid_df, data[-120:]])

    train_df.to_csv(directory_path + "train/train_df.csv")
    valid_df.to_csv(directory_path + "validation/valid_df.csv")

    return train_df, valid_df


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
def convert_and_cleanup(directory_path):
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
    csv_to_arrow(directory_path + "train/train_df.csv", directory_path + "train")
    # valid.csv arrow 파일로 변환
    csv_to_arrow(
        directory_path + "validation/valid_df.csv", directory_path + "validation"
    )

    # csv 파일, chunk.json 파일 제거
    delete_csv_json_in_aug_new(directory_path)


if __name__ == "__main__":
    directory_path = os.getenv("DIR_PATH") + "level2-mrc-nlp-11/data/aug_new/"
    output_path = os.getenv("DIR_PATH") + "level2-mrc-nlp-11/data/aug_new/aug_new.csv"

    # filter_unused_document()
    # create_new_document_qa()
    data = process_and_combine_csv(directory_path, output_path)
    answer_start_data = update_answer_start(data, directory_path)
    combine_train_valid(answer_start_data, directory_path)
    convert_and_cleanup(directory_path)
