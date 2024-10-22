import pandas as pd
import logging
import re

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

def main(input_csv_path, output_csv_path, log_file_path='error_log.txt'):
    """
    메인 함수: CSV 파일을 읽고, 처리하고, 결과를 저장합니다.

    Parameters:
        input_csv_path (str): 입력 CSV 파일 경로
        output_csv_path (str): 출력 CSV 파일 경로
        log_file_path (str): 로그 파일 경로
    """
    setup_logging(log_file_path)

    try:
        # CSV 파일 읽기
        df = pd.read_csv(input_csv_path, encoding='utf-8')
        logging.info(f"Successfully loaded input CSV file: {input_csv_path}")
    except Exception as e:
        logging.critical(f"Failed to read input CSV file: {e}")
        print(f"입력 CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 데이터 처리
    updated_df = update_answer_start(df)
    logging.info(f"Filtering complete. {len(updated_df)} out of {len(df)} items retained.")

    # 출력 CSV 파일로 저장
    try:
        updated_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"Successfully wrote output CSV file: {output_csv_path}")
        print(f"처리된 데이터를 '{output_csv_path}' 파일로 성공적으로 저장했습니다.")
    except Exception as e:
        logging.critical(f"Failed to write output CSV file: {e}")
        print(f"출력 CSV 파일을 쓰는 중 오류가 발생했습니다: {e}")

    # 로그 파일 안내
    print(f"오류 및 처리 과정을 확인하려면 '{log_file_path}' 파일을 참조하세요.")

if __name__ == "__main__":
    # 입력 및 출력 파일 경로 설정
    input_csv = '/AI-projects/ODQA-project/data/aug_prev/train/aug_prev_train_data.csv'        # 입력 CSV 파일 경로
    output_csv = '/AI-projects/ODQA-project/data/aug_prev/train/aug_prev_train_output_file.csv'      # 출력 CSV 파일 경로
    log_file = '/AI-projects/ODQA-project/data/aug_prev/train/error_log.txt'          # 로그 파일 경로

    main(input_csv, output_csv, log_file)
