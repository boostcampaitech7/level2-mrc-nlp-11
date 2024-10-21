import json
from datasets import load_from_disk
from dotenv import load_dotenv
import os
import asyncio
from call_openai_API import call_openai_API


def paraphrase_questions(questions, mode):

    # v1
    prompt_content = f"""
        다음은 질문 목록입니다. 각 질문 문장의 의미를 유지한채로 몇몇 단어를 동의어로 변경하여 패러프레이징해주세요.
        ex)
        - 올레 폰 보이스트 이후에 시장으로 임명된 사람은 누구인가? -> 올레 폰 보이스트 다음에 시장으로 선임된 인물은 누구인가?
        - 우메노의 사인은? -> 우메노가 죽은 원인은 무엇인가?
        응답은 원래 문장을 key, 바꾼 문장을 value로 하는 순수한 JSON 형식으로 제공해주세요. 단, 바꾼 문장에 코드 블록이나 다른 텍스트는 포함되지 않도록 해주세요. 또한, 원래 문장, 바꾼 문장에 "이 포함될 경우 이스케이프 문자를 붙여 JSON 파싱 과정에 오류가 생기지 않도록 해주세요.
        """

    # v2
    # prompt_content = f"""
    #     다음은 질문 목록입니다. 각 질문 문장의 의미를 유지한채로 패러프레이징해주세요. 일부 문장은 그대로, 일부 문장은 고유 명사를 제외한 몇몇 단어를 동의어로 대체, 일부 문장은 문법이 맞는 범위 내에서 어순을 바꿔주세요.
    #     ex)
    #     - 우메노의 사인은? -> 우메노가 죽은 원인은 무엇인가?
    #     - 올레 폰 보이스트 이후에 시장으로 임명된 사람은 누구인가? -> 누가 올레 폰 보이스트 다음 시장으로 선임되었는가?
    #     응답은 원래 문장을 key, 바꾼 문장을 value로 하는 순수한 JSON 형식으로 제공해주세요. 단, 바꾼 문장에 코드 블록이나 다른 텍스트는 포함되지 않도록 해주세요. 또한, 원래 문장, 바꾼 문장에 "이 포함될 경우 이스케이프 문자를 붙여 JSON 파싱 과정에 오류가 생기지 않도록 해주세요.
    #     """

    response_format = rf"""
        [
            {{"original": "올레 폰 보이스트 이후에 시장으로 임명된 사람은 누구인가?", "paraphrased": "누가 올레 폰 보이스트 다음 시장으로 선임되었는가?"}},
            {{"original": "2007년 자유 지수 순위 참여 국가 중 꼴찌를 한 국가는?", "paraphrased": "2007년 자유 지수 순위에 참여한 국가 중 최하위를 기록한 국가는?"}},
            ...
        ]
        """

    response = asyncio.run(
        call_openai_API(questions, 10, prompt_content, response_format)
    )

    paraphrased_questions = {}
    for batch in response:
        for paraphrased_question in batch:
            if paraphrased_question == None:
                continue
            paraphrased_questions[paraphrased_question["original"]] = (
                paraphrased_question["paraphrased"]
            )

    # save
    output_path = (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        + f"/data/paraphrased_{mode}_questions_v2.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(paraphrased_questions, f, ensure_ascii=False, indent=4)


def main():
    load_dotenv()

    dataset_path = os.getenv("DIR_PATH") + "/level2-mrc-nlp-11/data/default"
    dataset = load_from_disk(dataset_path)

    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    train_questions = train_dataset["question"]
    validation_questions = validation_dataset["question"]

    paraphrase_questions(train_questions, "train")
    paraphrase_questions(validation_questions, "validation")


if __name__ == "__main__":
    main()
