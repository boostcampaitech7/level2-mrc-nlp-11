import json
import logging
import re


def setup_logging(log_file="error_log.txt"):
    """
    로깅 설정을 초기화합니다.
    """
    logging.basicConfig(
        filename=log_file,
        filemode="w",  # 로그 파일을 덮어쓰기 모드로 설정
        level=logging.DEBUG,  # DEBUG 레벨 이상의 모든 로그를 기록
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def count_backslashes(text):
    # 백슬래시 하나를 찾는 정규표현식
    tokens = re.findall(r"\\", text)
    word_count = len(tokens)
    return word_count


def update_answer_start(data):
    """
    JSON 데이터에서 answers['text']가 context에 포함된 경우에만 필터링하고
    answer_start를 올바른 값으로 업데이트합니다.

    Parameters:
        data (list): JSON 객체의 리스트

    Returns:
        list: 필터링 및 업데이트된 JSON 객체의 리스트
    """
    filtered_data = []
    for idx, item in enumerate(data):
        context = item.get("context", "")
        answers = item.get("answers", {})
        answer_texts = answers.get("text", [])
        answer_starts = answers.get("answer_start", [])

        # 모든 answer_text가 context에 포함되어 있는지 확인
        all_present = True
        new_answer_starts = []
        breakpoint()
        for answer_text in answer_texts:
            start_idx = context.find(answer_text)
            if start_idx == -1:
                all_present = False
                logging.error(
                    f"Item ID {item.get('id')} - Answer text '{answer_text}' not found in context."
                )
                break
            start_idx += count_backslashes(context)
            new_answer_starts.append(start_idx)

        if all_present:
            # answer_start를 업데이트
            item["answers"]["answer_start"] = new_answer_starts
            filtered_data.append(item)
        else:
            logging.info(
                f"Item ID {item.get('id')} - Excluded due to missing answer text in context."
            )

    return filtered_data


def main(input_json_path, output_json_path, log_file_path="error_log.txt"):
    """
    메인 함수: JSON 파일을 읽고, 처리하고, 결과를 저장합니다.

    Parameters:
        input_json_path (str): 입력 JSON 파일 경로
        output_json_path (str): 출력 JSON 파일 경로
        log_file_path (str): 로그 파일 경로
    """
    setup_logging(log_file_path)

    try:
        # JSON 파일 읽기
        with open(input_json_path, "r", encoding="utf-8") as infile:
            data = json.load(infile)
            logging.info(f"Successfully loaded input JSON file: {input_json_path}")
    except Exception as e:
        logging.critical(f"Failed to read input JSON file: {e}")
        print(f"입력 JSON 파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 데이터 처리
    filtered_data = update_answer_start(data)
    logging.info(
        f"Filtering complete. {len(filtered_data)} out of {len(data)} items retained."
    )

    # 출력 JSON 파일로 저장
    try:
        with open(output_json_path, "w", encoding="utf-8") as outfile:
            json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)
            logging.info(f"Successfully wrote output JSON file: {output_json_path}")
        print(f"처리된 데이터를 '{output_json_path}' 파일로 성공적으로 저장했습니다.")
    except Exception as e:
        logging.critical(f"Failed to write output JSON file: {e}")
        print(f"출력 JSON 파일을 쓰는 중 오류가 발생했습니다: {e}")

    # 로그 파일 안내
    print(f"오류 및 처리 과정을 확인하려면 '{log_file_path}' 파일을 참조하세요.")


if __name__ == "__main__":
    # 입력 및 출력 파일 경로 설정
    input_json = "csv_to_json_output.json"  # 입력 JSON 파일 경로
    output_json = "filtered_output.json"  # 출력 JSON 파일 경로
    log_file = "error_log.txt"  # 로그 파일 경로

    main(input_json, output_json, log_file)
