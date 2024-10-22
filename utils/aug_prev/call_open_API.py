import pandas as pd
import asyncio
from datasets import Dataset
from qa_generator import QAGenerator


# 데이터 로드 및 저장 함수 정의
def load_data(path):
    dataset = Dataset.from_file(path)
    return pd.DataFrame(dataset)


def save_data(df, path):
    df.to_csv(path, index=False)

def save_data_as_arrow(df, path):
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(path)


async def main(train_path, valid_path, output_train_path, output_valid_path, output_train_arrow, output_valid_arrow):
    # 1. 데이터 로드
    train_data = load_data(train_path)
    valid_data = load_data(valid_path)

    # 2. QAGenerator 인스턴스 생성
    qa_generator = QAGenerator()

    # 3. 데이터 증강
    augmented_train_data = await qa_generator.augment_data(train_data)
    augmented_valid_data = await qa_generator.augment_data(valid_data)

    combined_train_data = pd.concat([train_data, augmented_train_data], ignore_index=True)
    combined_valid_data = pd.concat([valid_data, augmented_valid_data], ignore_index=True)

    # 4. 결과 저장
    save_data(combined_train_data, output_train_path)
    save_data(combined_valid_data, output_valid_path)
    save_data_as_arrow(combined_train_data, output_train_arrow)  # Arrow 형식으로 저장
    save_data_as_arrow(combined_valid_data, output_valid_arrow)  # Arrow 형식으로 저장
    print(f"증강된 훈련 데이터가 {output_train_path}에 저장되었습니다.")
    print(f"증강된 검증 데이터가 {output_valid_path}에 저장되었습니다.")


if __name__ == "__main__":

    train_path = "/data/ephemeral/home/level2-mrc-nlp-11/data/default/train/data-00000-of-00001.arrow"
    valid_path = "/data/ephemeral/home/level2-mrc-nlp-11/data/default/validation/data-00000-of-00001.arrow"
    output_train_path = "/data/ephemeral/home/level2-mrc-nlp-11/data/default/train/question_train_data.csv"
    output_valid_path = "/data/ephemeral/home/level2-mrc-nlp-11/data/default/validation/question_validation_data.csv"
    output_train_arrow = "/data/ephemeral/home/level2-mrc-nlp-11/data/default/train/question_train_data.arrow"
    output_valid_arrow = "/data/ephemeral/home/level2-mrc-nlp-11/data/default/validation/question_validation_data.arrow"

    asyncio.run(main(train_path, valid_path, output_train_path, output_valid_path, output_train_arrow, output_valid_arrow))
