import json, os
from dotenv import load_dotenv
from konlpy.tag import *

load_dotenv()


def get_wiki_morphs(
    analyzer_type="kkma",
    wiki_file_path=os.getenv("DIR_PATH")
    + "/level2-mrc-nlp-11/data/wikipedia_documents.json",
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
        f"{wiki_file_path.replace('.json', '')}_{analyzer_type}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(morphs_data, f, ensure_ascii=False, indent=4)


def remove_meaningless_kkma_moprhs(
    kkma_morphs_file_path=os.getenv("DIR_PATH")
    + "/level2-mrc-nlp-11/data/wikipedia_documents_kkma.json",
):
    josa_tag_list = ["JKS", "JKC", "JKG", "JKO", "JKM", "JKI", "JKQ", "JC", "JX"]
    suffix_list = (
        ["EPH", "EPT", "EPP", "EFN"]
        + ["EFQ", "EFO", "EFA", "EFI"]
        + ["EFR", "ECE", "ECS", "ECD"]
        + ["ETN", "ETD", "XSN", "XSV", "XSA"]
    )
    etc_list = ["IC", "SF", "SE", "SS", "SP", "SO"]
    tag_set = set(josa_tag_list + suffix_list + etc_list)

    with open(kkma_morphs_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    pos_data = {}
    for idx, (key, value) in enumerate(data.items()):
        filtered_pos_text = []
        for token, pos in value["pos_text"]:
            if pos in tag_set:
                continue
            filtered_pos_text.append(token)
        value["filtered_pos_text"] = filtered_pos_text
        pos_data[key] = value
        print(idx)
        idx += 1

    with open(
        f"{kkma_morphs_file_path.replace('.json', '')}_filtered.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(pos_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    get_wiki_morphs()
