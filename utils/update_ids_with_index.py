import pandas as pd

def update_ids_with_index(csv_file_path, output_csv_file_path):
    # CSV 파일을 읽어 DataFrame으로 변환
    data = pd.read_csv(csv_file_path)

    # 'id' 열에 인덱스를 추가
    data['id'] = data['id'].astype(str) + "_" + data.index.astype(str)

    # 업데이트된 DataFrame을 새로운 CSV 파일로 저장
    data.to_csv(output_csv_file_path, index=False)

    print(f"Updated IDs saved to {output_csv_file_path}")

# 사용 예시
input_csv_file = '/AI-projects/ODQA-project/data/aug_prev/aug_prev_train_output_file.csv'  # 입력 CSV 파일 경로
output_csv_file = '/AI-projects/ODQA-project/data/aug_prev/train/aug_prev_train_update_id.csv'  # 출력 CSV 파일 경로
update_ids_with_index(input_csv_file, output_csv_file)