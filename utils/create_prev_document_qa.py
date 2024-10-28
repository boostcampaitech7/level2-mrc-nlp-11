import json
import pandas as pd
from datasets import load_from_disk
import asyncio
from call_openai_API import call_openai_API
import os
from dotenv import load_dotenv
import ast
from datasets import Dataset, Features, Sequence, Value
import logging
import re


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

# openai api를 활용하여 기존 문서로부터 새로운 질문-답변 쌍(5개) 생성하는 함수
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


def setup_logging(log_file='error_log.txt'):
    """
    로깅 설정을 초기화합니다.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='w',  # 로그 파일을 덮어쓰기 모드로 설정
        level=logging.DEBUG,  # DEBUG 레벨 이상의 모든 로그를 기록
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def count_escape_chars(text):
    """
    문자열에서 이스케이프가 필요한 문자의 개수를 반환합니다.
    
    Args:
        text (str): 검사할 문자열
    
    Returns:
        int: 이스케이프가 필요한 문자의 수
    """
    pattern = r'["\\\b\f\n\r\t]|[\x00-\x1F]'  # 큰따옴표, 역슬래시, 제어 문자
    return len(re.findall(pattern, text))

# CSV 파일 불러오는 함수
def read_csv_files():
    train = pd.read_csv(os.getenv('DIR_PATH') + "level2-mrc-nlp-11/data/aug_prev/aug_prev_train.csv")
    valid = pd.read_csv(os.getenv('DIR_PATH') + "level2-mrc-nlp-11/data/aug_prev/aug_prev_valid.csv")

    return train, valid

# answer_start 값을 업데이트하는 함수
def update_answer_start(df):
    """
    DataFrame에서 답변의 시작 위치를 업데이트합니다.
    
    Args:
        df (DataFrame): 처리할 DataFrame
    
    Returns:
        DataFrame: 업데이트된 DataFrame
    """
    filtered_data = []
    for idx, row in df.iterrows():
        context = row['context']
        answers = eval(row['answers'])  # 문자열을 딕셔너리로 변환
        answer_texts = answers.get('text', [])
        answer_starts = answers.get('answer_start', [])

        all_present = True
        new_answer_starts = []

        for answer_text in answer_texts:
            start_idx = context.find(answer_text)
            if start_idx == -1:
                all_present = False
                logging.error(f"Item ID {row['id']} - Answer text '{answer_text}' not found in context.")
                break
            start_idx += count_escape_chars(context[:start_idx])
            new_answer_starts.append(start_idx)

        if all_present:
            answers['answer_start'] = new_answer_starts  # 업데이트된 시작 위치
            row['answers'] = answers  # 업데이트된 답변을 원래 DataFrame에 반영
            filtered_data.append(row)
        else:
            logging.info(f"Item ID {row['id']} - Excluded due to missing answer text in context.")

    return pd.DataFrame(filtered_data)

# answer_start 값 업데이트 후 csv 파일로 저장하는 함수
def process_answers_and_save():
    setup_logging('error_log.txt')
    try:
        # CSV 파일 읽기
        aug_train, aug_valid = read_csv_files()
        logging.info(f"Successfully loaded input CSV file")
    except Exception as e:
        logging.critical(f"Failed to read input CSV file: {e}")
        print(f"입력 CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return
    
    # 데이터 처리: DataFrame의 answer_start 필드를 수정합니다
    updated_train = update_answer_start(aug_train)
    updated_valid = update_answer_start(aug_valid)
    logging.info(f"Filtering complete. {len(updated_train)} out of {len(aug_train)} items retained.")
    logging.info(f"Filtering complete. {len(updated_valid)} out of {len(aug_valid)} items retained.")

    # 수정된 DataFrame을 새로운 CSV 파일로 저장합니다
    try:
        updated_train.to_csv(os.getenv('DIR_PATH') + "level2-mrc-nlp-11/data/aug_prev/aug_prev_train.csv", index=False, encoding='utf-8-sig')
        updated_valid.to_csv(os.getenv('DIR_PATH') + "level2-mrc-nlp-11/data/aug_prev/aug_prev_valid.csv", index=False, encoding='utf-8-sig')
        logging.info(f"Successfully wrote output CSV file")
        print(f"처리된 데이터를 성공적으로 저장했습니다.")
    except Exception as e:
        logging.critical(f"Failed to write output CSV file: {e}")
        print(f"출력 CSV 파일을 쓰는 중 오류가 발생했습니다: {e}")

    # 로그 파일 안내
    print(f"오류 및 처리 과정을 확인하려면 'error_log.txt' 파일을 참조하세요.")

# id 값 업데이트한 후 저장하는 함수
def update_ids_with_index():
    # CSV 파일을 읽어 DataFrame으로 변환
    train_data, valid_data = read_csv_files()
    output_train_path = pd.read_csv(os.getenv('DIR_PATH') + "level2-mrc-nlp-11/data/aug_prev/aug_prev_train.csv")
    output_valid_path = pd.read_csv(os.getenv('DIR_PATH') + "level2-mrc-nlp-11/data/aug_prev/aug_prev_valid.csv")

    # 'id' 열에 인덱스를 추가
    train_data['id'] = train_data['id'].astype(str) + "_" + train_data.index.astype(str)
    valid_data['id'] = valid_data['id'].astype(str) + "_" + valid_data.index.astype(str)

    # 업데이트된 DataFrame을 새로운 CSV 파일로 저장
    train_data.to_csv(output_train_path, index=False)
    valid_data.to_csv(output_valid_path, index=False)

    print(f"Updated IDs saved to {output_train_path}")
    print(f"Updated IDs saved to {output_valid_path}")


def csv_to_arrow():
    # Step 1: Read the CSV file
    train_data, valid_data = read_csv_files()
    valid_data = valid_data[240:]

    # Step 2: Combine train_data and valid_data
    combined_data = pd.concat([train_data, valid_data], ignore_index=True)
    # DataFrame을 랜덤하게 섞습니다.
    combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)  # frac=1은 전체 데이터셋을 섞음

    # Step 3: Parse the 'answers' column
    combined_data["answers"] = combined_data["answers"].apply(ast.literal_eval)

    # Step 4: Ensure the 'answers' column has the correct structure
    combined_data["answers"] = combined_data["answers"].apply(lambda x: {
        'answer_start': x['answer_start'],  # 리스트 형태 유지
        'text': x['text']
    })

    # Debug: Print the first few rows of the 'answers' column
    print(combined_data["answers"].head())


    # Step 4: Define the features schema
    features = Features(
        {
            "id": Value("string"),
            "title": Value("string"),
            "context": Value("string"),
            "question": Value("string"),
            "answers": {
                "answer_start": Sequence(Value("int64")),  # Ensure int64 is used
                "text": Sequence(Value("string")),
            },
            "document_id": Value("int64"),               # document_id 추가
            "__index_level_0__": Value("int64")         # __index_level_0__ 추가
        }
    )

    # Step 6: Create the Dataset
    try:
        dataset = Dataset.from_pandas(combined_data, features=features)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"DataFrame structure: {combined_data.dtypes}")
        print(combined_data.head())
        return  # Early exit on error

    # Step 7: Save the Dataset to disk
    output_path = os.getenv('DIR_PATH') + "level2-mrc-nlp-11/data/aug_prev/train"
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")



if __name__ == "__main__":
    create_prev_document_qa()
    process_answers_and_save()
    update_ids_with_index()
    csv_to_arrow()

# mrc.yaml 파일의 dataset 이름 aug_prev로 바꾸기
# ./run_mrc.sh 돌려서 확인하기
