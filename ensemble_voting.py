import json
import os
import sys
from collections import defaultdict, Counter, OrderedDict
from typing import List, Dict, Any, Union

# ---------------- 사용자 입력 부분 ----------------
# 보팅 방법 선택: 'soft', 'hard', 'weighted-soft'
voting_method = "hard"  # 'soft', 'hard', 'weighted-soft' 중 선택

# 앙상블할 JSON 파일들의 경로를 리스트로 지정합니다.
input_files = [
    "/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/outputs1/sep50_sp-mo-dense-17-4-1_original_default_bz=16_lr=1.711_test_predictions(1).json",
    "/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/outputs1/sep-inf50_rerank85_original_default_bz=16_lr=1.7110631470130408e-05_test_predictions.json",
    "/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/outputs1/sep50_sp-com-dense842_original_default_bz=16_lr=1.711_test_predictions.json",
    "/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/outputs1/sep50_sp-mo-dense842_original_default_bz=16_lr=1.7110631470130408e-05_test_predictions.json",
    "/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/outputs1/sep100_rerank85_original_default_bz=16_lr=1.7110631470130408e-05_test_predictions.json",
    "/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/outputs1/sep40_rerank85_original_default_bz=16_lr=1.711_test_predictions.json",
    "/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/outputs1/sep70_rerank88_original_default_bz=16_lr=1.711_test_predictions.json",
    # 필요한 만큼 추가하세요.
]

# 결과를 저장할 출력 파일의 경로를 지정합니다.
output_file = f"{voting_method}_ensemble_results_4.json"

# 가중치를 지정합니다 (weighted-soft 방식인 경우).
# 각 가중치는 input_files의 모델 순서와 동일한 순서로 지정됩니다.
weights = [0.6, 0.4]
# ---------------------------------------------------


def load_json_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    여러 JSON 파일을 로드하여 리스트로 반환합니다.
    """
    json_data_list = []
    file_types = []  # 각 파일의 형식 (1: 단순 답변, 2: 예측 리스트)

    for file_path in file_paths:
        if not os.path.isfile(file_path):
            print(f"파일을 찾을 수 없습니다: {file_path}")
            sys.exit(1)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            json_data_list.append(data)
            # 파일 형식 감지
            first_value = next(iter(data.values()))
            if isinstance(first_value, list):
                file_types.append(2)
            else:
                file_types.append(1)
    return json_data_list, file_types


def ensemble_soft_voting(
    json_data_list: List[Dict[str, Any]], file_types: List[int]
) -> OrderedDict:
    """
    소프트 보팅 앙상블을 수행하여 각 키에 대해 최종 선택된 텍스트를 반환합니다.
    """
    soft_voting_results = OrderedDict()

    # 모든 키 수집 (모든 파일에서 키를 수집하여 합집합 생성)
    all_keys = set()
    for data in json_data_list:
        all_keys.update(data.keys())

    # 키 순서를 첫 번째 파일의 순서로 유지
    keys = list(json_data_list[0].keys())

    for key in keys:
        # 예측을 집계할 딕셔너리: (text) -> [확률 리스트]
        prediction_dict = defaultdict(list)

        for idx, json_data in enumerate(json_data_list):
            data = json_data.get(key)
            if data is None:
                continue
            file_type = file_types[idx]
            if file_type == 1:
                # 1번 형식: 단순 답변
                pred_key = json_data[key]
                probability = 1.0  # 기본 확률 부여
                prediction_dict[pred_key].append(probability)
            elif file_type == 2:
                # 2번 형식: 예측 리스트
                predictions = data
                for pred in predictions:
                    pred_key = pred["text"]
                    probability = pred.get("probability", 0)
                    prediction_dict[pred_key].append(probability)
            else:
                continue

        if not prediction_dict:
            continue

        # 평균 확률을 계산하고 가장 높은 평균 확률을 가진 예측을 선택합니다.
        best_prediction = None
        highest_avg_prob = -1.0

        for pred_key, prob_list in prediction_dict.items():
            avg_prob = sum(prob_list) / len(prob_list)
            if avg_prob > highest_avg_prob:
                highest_avg_prob = avg_prob
                best_prediction = pred_key

        if best_prediction:
            soft_voting_results[key] = best_prediction  # 텍스트만 저장

    return soft_voting_results


def ensemble_weighted_soft_voting(
    json_data_list: List[Dict[str, Any]], file_types: List[int], weights: List[float]
) -> OrderedDict:
    """
    가중치를 사용하여 소프트 보팅 앙상블을 수행하여 각 키에 대해 최종 선택된 텍스트를 반환합니다.
    """
    weighted_soft_voting_results = OrderedDict()

    # 모든 키 수집 (모든 파일에서 키를 수집하여 합집합 생성)
    all_keys = set()
    for data in json_data_list:
        all_keys.update(data.keys())

    # 키 순서를 첫 번째 파일의 순서로 유지
    keys = list(json_data_list[0].keys())

    for key in keys:
        # 예측을 집계할 딕셔너리: (text) -> [가중치가 적용된 확률의 합, 가중치 합]
        prediction_dict = defaultdict(lambda: [0.0, 0.0])

        for idx, json_data in enumerate(json_data_list):
            data = json_data.get(key)
            if data is None:
                continue
            weight = weights[idx]
            file_type = file_types[idx]
            if file_type == 1:
                # 1번 형식: 단순 답변
                pred_key = json_data[key]
                probability = 1.0  # 기본 확률 부여
                prediction_dict[pred_key][0] += probability * weight
                prediction_dict[pred_key][1] += weight
            elif file_type == 2:
                # 2번 형식: 예측 리스트
                predictions = data
                for pred in predictions:
                    pred_key = pred["text"]
                    probability = pred.get("probability", 0)
                    prediction_dict[pred_key][0] += probability * weight
                    prediction_dict[pred_key][1] += weight
            else:
                continue

        if not prediction_dict:
            continue

        # 가중치가 적용된 평균 확률을 계산하고 가장 높은 값을 가진 예측을 선택합니다.
        best_prediction = None
        highest_weighted_avg_prob = -1.0

        for pred_key, (weighted_sum, total_weight) in prediction_dict.items():
            weighted_avg_prob = weighted_sum / total_weight if total_weight != 0 else 0
            if weighted_avg_prob > highest_weighted_avg_prob:
                highest_weighted_avg_prob = weighted_avg_prob
                best_prediction = pred_key

        if best_prediction:
            weighted_soft_voting_results[key] = best_prediction  # 텍스트만 저장

    return weighted_soft_voting_results


def ensemble_hard_voting(
    json_data_list: List[Dict[str, Any]], file_types: List[int]
) -> OrderedDict:
    """
    하드 보팅 앙상블을 수행하여 각 키에 대해 최종 선택된 텍스트를 반환합니다.
    """
    hard_voting_results = OrderedDict()

    # 모든 키 수집 (모든 파일에서 키를 수집하여 합집합 생성)
    all_keys = set()
    for data in json_data_list:
        all_keys.update(data.keys())

    # 키 순서를 첫 번째 파일의 순서로 유지
    keys = list(json_data_list[0].keys())

    for key in keys:
        # 예측을 집계할 Counter: (text) -> 투표 수
        prediction_counter = Counter()

        for idx, json_data in enumerate(json_data_list):
            data = json_data.get(key)
            if data is None:
                continue
            file_type = file_types[idx]
            if file_type == 1:
                # 1번 형식: 단순 답변
                pred_key = json_data[key]
                prediction_counter[pred_key] += 1
            elif file_type == 2:
                # 2번 형식: 예측 리스트
                predictions = data
                for pred in predictions:
                    pred_key = pred["text"]
                    prediction_counter[pred_key] += 1
            else:
                continue

        if not prediction_counter:
            continue

        # 가장 많은 투표를 받은 예측을 선택합니다.
        best_prediction_key, _ = prediction_counter.most_common(1)[0]
        hard_voting_results[key] = best_prediction_key  # 텍스트만 저장

    return hard_voting_results


def save_ensemble_results(ensemble_results: OrderedDict, output_path: str):
    """
    앙상블된 결과를 JSON 파일로 저장합니다.
    """
    with open("./outputs/" + output_path, "w", encoding="utf-8") as f:
        json.dump(ensemble_results, f, ensure_ascii=False, indent=4)


def main():
    # JSON 파일들을 로드합니다.
    json_data_list, file_types = load_json_files(input_files)

    if voting_method == "soft":
        # 소프트 보팅 앙상블을 수행합니다.
        ensemble_results = ensemble_soft_voting(json_data_list, file_types)
    elif voting_method == "hard":
        # 하드 보팅 앙상블을 수행합니다.
        ensemble_results = ensemble_hard_voting(json_data_list, file_types)
    elif voting_method == "weighted-soft":
        # 가중치 리스트의 길이가 입력 파일의 수와 동일한지 확인합니다.
        if not weights or len(weights) != len(input_files):
            print("가중치의 수는 입력 파일의 수와 동일해야 합니다.")
            sys.exit(1)
        # 가중치의 합이 0인지 확인합니다.
        if sum(weights) == 0:
            print("가중치의 합은 0이 될 수 없습니다.")
            sys.exit(1)
        ensemble_results = ensemble_weighted_soft_voting(
            json_data_list, file_types, weights
        )
    else:
        print("올바른 보팅 방법을 선택해주세요: soft, hard, weighted-soft")
        sys.exit(1)

    # 결과를 저장합니다.
    save_ensemble_results(ensemble_results, output_file)

    print(
        f"{voting_method.capitalize()} 보팅 앙상블된 결과가 저장되었습니다: {output_file}"
    )


if __name__ == "__main__":
    main()
