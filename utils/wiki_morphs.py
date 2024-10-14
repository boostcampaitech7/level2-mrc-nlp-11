import json
import pandas as pd
from konlpy.tag import *


def get_wiki_morphs(
    analyzer_type="kkma", wiki_file_path="data/wikipedia_documents.json"
):

    # 1. 형태소 분석기 선택
    if analyzer_type == "kkma":
        analyzer = Kkma()
    elif analyzer_type == "okt":
        analyzer = Okt()
    elif analyzer_type == "komoran":
        analyzer = Komoran()
    elif analyzer_type == "hananum":
        analyzer = Hannanum()
    elif analyzer_type == "mecab":
        analyzer = Mecab()
    else:
        print(f"{analyzer_type}은 없습니다.")
        return

    # 2. wiki 문서 load
    with open(wiki_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 3. 형태소 분석
    morphs_data = {}
    for idx, (key, value) in enumerate(data.items()):
        try:
            pos_text = analyzer.pos(value["text"])
        except Exception:
            continue
        print(idx)
        value["pos_text"] = pos_text
        morphs_data[key] = value

    # 4. 파일 저장
    with open(
        f"./data/wikipedia_documents_{analyzer_type}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(morphs_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    get_wiki_morphs()
