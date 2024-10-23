import os
import json
import re
import asyncio
from call_openai_API import call_openai_API


def categorize_taglike_words(wiki_documents):
    wiki_texts = [wiki_document["text"] for wiki_document in wiki_documents.values()]

    def find_taglike_words(text):
        pattern = r"<(?![^<>]*[가-힣])[^<>]+>"
        matches = re.findall(pattern, text)
        return matches

    taglike_word_list = []
    for text in wiki_texts:
        taglike_words = find_taglike_words(text)
        if taglike_words:
            taglike_word_list.extend(taglike_words)
    taglike_word_list = list(set(taglike_word_list))

    prompt_content = f"""
        다음은 100개의 태그 형식 문자열(<> 사이에 문자열이 있는 형태)입니다. 각 문자열을 다음 세 가지 카테고리 중 하나로 분류해주세요: HTML_TAG, CODE_REPRESENTATION, PROPER_NOUN.
        응답은 문자열을 key, 카테고리를 value로 하는 순수한 JSON 형식으로 제공해주세요. 코드 블록이나 다른 텍스트는 포함하지 마세요.
        """

    response_format = f"""
        [
            {{"string1": "HTML_TAG"}},
            {{"string2": "PROPER_NOUN"}},
            ...
        ]
        """

    response = asyncio.run(
        call_openai_API(taglike_word_list, 100, prompt_content, response_format)
    )

    categorized_results = {}
    for categorized_words in response:
        for categorized_word in categorized_words:
            if categorized_word == None:
                continue
            for word, category in categorized_word.items():
                categorized_results[word] = category

    # save
    output_path = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + "/data/categorized_taglike_words.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(categorized_results, f, ensure_ascii=False, indent=4)
    return categorized_results


def remove_code_documents(taglike_words_category, wiki_documents):
    code_set = set()
    for taglike_word, category in taglike_words_category.items():
        if category == "CODE_REPRESENTATION":
            for doc_id, doc in wiki_documents.items():
                if taglike_word in doc["text"]:
                    code_set.add(doc_id)

    # 수기로 검증하여 의미가 있다고 판단한 문서(문서 전체가 코드로 도배되어 있지 않은 문서)
    for idx in ["43445", "34273", "13921", "59622", "15290", "55000", "28801", "41935"]:
        if idx in code_set:
            code_set.remove(idx)

    code_list = list(code_set)
    for doc_id in code_list:
        del wiki_documents[doc_id]

    code_set = set()
    code_pattern = r"(\+=|\\frac|return |stdio|[^C]\+\+| int |def |const )"
    for doc in wiki_documents.values():
        codelike = re.findall(code_pattern, doc["text"])
        if codelike:
            code_set.update(codelike)

    # 문서 삭제 또는 전처리가 필요한 문서를 수기 분류
    code_delete = [
        "1164",
        "1254",
        "1927",
        "2638",
        "4152",
        "5718",
        "5808",
        "6481",
        "7192",
        "8706",
        "9193",
        "12104",
        "12700",
        "12702",
        "12703",
        "12704",
        "15291",
        "16070",
        "17096",
        "17097",
        "21403",
        "25097",
        "32382",
        "37378",
        "45570",
        "45705",
        "46851",
        "48850",
        "51530",
        "55062",
        "55063",
        "55064",
        "55065",
        "55066",
        "55068",
        "55069",
        "55070",
        "59622",
        "49452",
        "54545",
        "23451",
        "60",
        "26706",
    ]
    code_preprocess = [
        "362",
        "1255",
        "3868",
        "4916",
        "5809",
        "7394",
        "8422",
        "9384",
        "11668",
        "15292",
        "34273",
        "34292",
        "41663",
        "45189",
        "49013",
        "50502",
        "50992",
        "54440",
        "59474",
        "59689",
        "59690",
        "13630",
    ]

    for code_id in code_delete:
        wiki_documents.pop(code_id, None)

    documents_with_code = []
    for doc_id in code_preprocess:
        if doc_id in wiki_documents.keys():
            documents_with_code.append((doc_id, wiki_documents[doc_id]["text"]))

    prompt_content = f"""
        다음은 1개의 문서 목록입니다. 이 문서들은 위키피디아 문서의 일부를 크롤링한 문서들이며, 코드 표현 또는 latex 수식 표현이 포함되어 있습니다. 예) 'nunsigned int fib(unsigned int n)', 'main(int argc, char *argv[])', 'integer p = rq(x) + rq(y)'
        각 문서에서 이러한 긴 코드 표현 및 수식 표현을 제거하여 반환해주세요. 단, '할당 : malloc() : 힙영역으로 부터 데이터 공간을 할당 받는다.'에서 'malloc()'과 같이 용어 설명이나 짧은 인라인 코드인 경우는 그대로 유지해주세요.
        응답은 문서 번호와 코드 표현 및 수식 표현을 제거한 문서를 순수한 JSON 형식으로 제공해주세요. 단, 'Invalid \escape' 오류가 발생하지 않도록 텍스트에 이스케이프 문자가 포함된 경우 raw string 처리하여 문자 그대로 취급되게 반환해주세요. 코드 블록이나 다른 텍스트는 포함하지 마세요.
        """

    response_format = f"""
        [
        {{"document_id": 1, "preprocessed_text": "전처리된 텍스트"}},
        {{"document_id": 3, "preprocessed_text": "코드 표현이 제거된 텍스트입니다."]}},
        ...
        ]
        """

    response = asyncio.run(
        call_openai_API(documents_with_code, 1, prompt_content, response_format)
    )

    preprocessed_documents = {}
    for docs in response:
        for doc in docs:
            if not doc:
                continue
            preprocessed_documents[doc["document_id"]] = {
                "text": doc["preprocessed_text"]
            }

    # save
    save_path = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + "/data/documents_removed_code.json"
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(preprocessed_documents, f, ensure_ascii=False, indent=4)

    # 전처리된 문서 입력
    for doc_id, preprocessed_text in preprocessed_documents.items():
        wiki_documents[doc_id]["text"] = preprocessed_text["text"]

    return wiki_documents


def remove_HTML_tag(taglike_words_category, wiki_documents):
    html_tag_set = set()

    for taglike_word, category in taglike_words_category.items():
        if category == "HTML_TAG":
            html_tag_set.add(taglike_word)
            for doc_id, doc in wiki_documents.items():
                if taglike_word in doc["text"]:
                    processed_text = doc["text"]
                    processed_text = re.sub(taglike_word, " ", processed_text)
                    wiki_documents[doc_id]["text"] = processed_text

    return wiki_documents


def remove_page_reference(wiki_documents):
    for doc_id, doc in wiki_documents.items():
        page_refs = re.findall(r"(?:p{1,2}|pages?)=[0-9]+(?:[–-][0-9]+)?", doc["text"])
        if not page_refs:
            continue
        processed_text = doc["text"]
        for page_ref in page_refs:
            # 연도 표시와 붙어있는 경우 3가지 존재 -> 직접 문자열 수정
            if page_ref == "p=3031818":
                page_ref = "p=303"
            elif page_ref == "pages=443–4452015":
                page_ref = "pages=443–445"
            elif page_ref == "p=451938":
                page_ref = "p=45"
            processed_text = re.sub(
                r"(?:p{1,2}|pages?)=[0-9]+(?:[–-][0-9]+)?", " ", processed_text
            )
        wiki_documents[doc_id]["text"] = processed_text

    return wiki_documents


def remove_url(wiki_documents):
    url_pattern = r"https?:/?/?[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})(?:/[^\s]*)?"
    for doc_id, doc in wiki_documents.items():
        urls = re.findall(url_pattern, doc["text"])
        if urls:
            processed_text = doc["text"]
            processed_text = re.sub(url_pattern, " ", processed_text)
            wiki_documents[doc_id]["text"] = processed_text

    return wiki_documents


def remove_loc_reference(wiki_documents):
    def find_loc_ref(text):
        refs = []
        refs.extend(re.findall(r"loc=[0-9]+ and [0-9]+", text))
        text = re.sub(r"loc=[0-9]+ and [0-9]+", " ", text)

        refs.extend(re.findall(r"loc=§ [0-9]+ and [0-9]+", text))
        text = re.sub(r"loc=§ [0-9]+ and [0-9]+", " ", text)

        refs.extend(re.findall(r"loc=[0-9]+", text))
        text = re.sub(r"loc=[0-9]+", " ", text)

        refs.extend(re.findall(r"loc=§ [0-9]+", text))
        text = re.sub(r"loc=§ [0-9]+", " ", text)

        refs.extend(re.findall(r"loc=[a-zA-Z0-9 ]+", text))
        text = re.sub(r"loc=[a-zA-Z0-9 ]+", " ", text)

        refs.extend(re.findall(r"loc=[^ .]*", text))
        text = re.sub(r"loc=[^ .]*", " ", text)
        return text, refs

    for doc_id, doc in wiki_documents.items():
        processed_text = doc["text"]
        processed_text, locs = find_loc_ref(processed_text)
        if locs:
            wiki_documents[doc_id]["text"] = processed_text

    return wiki_documents


def remove_annotation(wiki_documents):
    for doc_id, doc in wiki_documents.items():
        annotations = re.findall(r"\|[\s\S]*?\}\}", doc["text"])
        if annotations:
            processed_text = doc["text"]
            processed_text = re.sub(r"\|[\s\S]*?\}\}", " ", processed_text)
            wiki_documents[doc_id]["text"] = processed_text

    need_remove = [
        "973",
        "1667",
        "1970",
        "3780",
        "3879",
        "4379",
        "5527",
        "6221",
        "8334",
        "8433",
        "8933",
        "9335",
        "9693",
        "10156",
        "11949",
        "13192",
        "14412",
        "22519",
        "26045",
        "32847",
        "39948",
        "40095",
        "41224",
        "43406",
        "56572",
        "59707",
    ]
    for doc_id in need_remove:
        wiki_documents.pop(doc_id, None)

    need_normalize = ["1945", "1824", "2016", "7527", "28172", "37746", "39067"]
    for doc_id in need_normalize:
        if doc_id in wiki_documents.keys():
            wiki_documents[doc_id]["text"] = re.sub(
                r"[^ ]*\|[^|]+\|(?:[^|]+\|)*[^ \n]*",
                " ",
                wiki_documents[doc_id]["text"],
            )

    # |string| 형태의 주석
    documents_with_annotation = {}
    for doc_id, doc in wiki_documents.items():
        annotations = re.findall(r"[^ ]*\|[^|]+\|(?:[^|]+\|)*[^ ]*", doc["text"])
        if not annotations:
            continue
        documents_with_annotation[doc_id] = doc["text"]

    # save
    annotation_documents = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + "/data/documents_with_annotation.json"
    )
    with open(annotation_documents, "w", encoding="utf-8") as writer:
        writer.write(
            json.dumps(documents_with_annotation, indent=4, ensure_ascii=False) + "\n"
        )

    annotation_documents_list = [
        (doc_id, text) for doc_id, text in annotation_documents.items()
    ]

    prompt_content = f"""
        다음은 1개의 문서 목록입니다. 이 문서들은 위키피디아 문서의 일부를 크롤링한 문서들이며, 위키피디아 문법 주석이 포함되어 있습니다. 예) 'type=music|pos=left|filename=Ludwig van Beethoven - Symphonie 5 c-moll - 1. Allegro con brio.ogg|title=교향곡 5번, 작품번호 67 (1악장)|description=', '37.939722|27.340833|name=에페소스|type:city|display=title|format=dms'
        각 문서에서 이러한 주석을 제거하고, 제거한 주석 문자열의 리스트와 함께 반환해주세요.
        응답은 문서 번호와 주석을 제거한 문서, 제거한 주석 문자열 리스트를 순수한 JSON 형식으로 제공해주세요. 단, 'Invalid \escape' 오류가 발생하지 않도록 텍스트에 이스케이프 문자가 포함된 경우 raw string 처리하여 문자 그대로 취급되게 반환해주세요. 코드 블록이나 다른 텍스트는 포함하지 마세요.
        """

    response_format = f"""
        [
        {{"document_id": 1, "preprocessed_text": "주석이 제거된 텍스트", "removed_annotations": ["annotation1", "annotation2"]}},
        {{"document_id": 3, "preprocessed_text": "주석이 제거된 텍스트입니다.", "removed_annotations": ["annotation3", "annotation4", "annotation5"]}},
        ...
        ]
        """

    # openai API 호출: 주석 문구 전처리
    response = asyncio.run(
        call_openai_API(annotation_documents_list, 1, prompt_content, response_format)
    )

    preprocessed_documents = {}
    for docs in response:
        for doc in docs:
            continue
        preprocessed_documents[doc["document_id"]] = {
            "text": doc["preprocessed_text"],
            "removed": doc["removed_annotations"],
        }

    # save
    save_path = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + "/data/documents_removed_annotation.json"
    )
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(preprocessed_documents, f, ensure_ascii=False, indent=4)

    # 적용
    for doc_id, preprocessed_doc in preprocessed_documents.items():
        if doc_id in wiki_documents.keys():
            wiki_documents[doc_id]["text"] = preprocessed_doc["text"]

    return wiki_documents


def main():
    wiki_path = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + "/data/wikipedia_documents.json"
    )
    if os.path.exists(wiki_path):
        with open(wiki_path, "r", encoding="utf-8") as f:
            wiki_documents = json.load(f)

    categorized_taglike_words = categorize_taglike_words(wiki_documents)
    wiki_documents = remove_code_documents(categorized_taglike_words, wiki_documents)
    wiki_documents = remove_HTML_tag(categorized_taglike_words, wiki_documents)
    wiki_documents = remove_page_reference(wiki_documents)
    wiki_documents = remove_url(wiki_documents)
    wiki_documents = remove_loc_reference(wiki_documents)
    wiki_documents = remove_annotation(wiki_documents)

    # save
    normalized_wikipedia_documents_path = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + "/data/wikipedia_documents.json"
    )
    with open(normalized_wikipedia_documents_path, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(wiki_documents, indent=4, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
