import sys, os, requests, tarfile, shutil, pickle, copy, json, glob, re
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    DatasetDict,
    Sequence,
    Features,
    Value,
    concatenate_datasets,
)

"""
STANDARD DATA FORMAT
{
    'id': Value(dtype='string', id=None),
    'title': Value(dtype='string', id=None),
    'context': Value(dtype='string', id=None),
    'question': Value(dtype='string', id=None),
    'answers': Sequence(feature={
        'text': Value(dtype='string', id=None),
        'answer_start': Value(dtype='int32', id=None)
    }, length=-1, id=None)
}
"""


def get_standard_features():
    return Features(
        {
            "id": Value("string"),
            "title": Value("string"),
            "context": Value("string"),
            "question": Value("string"),
            "answers": Sequence(
                {"text": Value("string"), "answer_start": Value("int32")}
            ),
        }
    )


def get_dataset_list(dataset_name_list):
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_list = []
    for dataset_name in dataset_name_list:
        if os.path.exists(f"{parent_directory}/data/{dataset_name}"):
            dataset_list.append(
                load_from_disk(f"{parent_directory}/data/{dataset_name}")
            )
        else:
            # download dataset and formatting
            current_module = sys.modules[__name__]
            getattr(current_module, dataset_name)()
            dataset_list.append(
                load_from_disk(f"{parent_directory}/data/{dataset_name}")
            )
    return dataset_list


def default():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    url = "https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000328/data/data.tar.gz"
    file_name = f"{current_directory}/data.tar.gz"

    # 1. Download dataset
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_name, "wb") as f:
            f.write(response.raw.read())

    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(path=current_directory)

    # 2. move wiki docs file and test dataset file to data directory
    if not os.path.exists(f"{parent_directory}/data/"):
        os.makedirs(f"{parent_directory}/data/")
    shutil.move(
        f"{current_directory}/data/wikipedia_documents.json",
        f"{parent_directory}/data/wikipedia_documents.json",
    )
    shutil.move(
        f"{current_directory}/data/test_dataset",
        f"{parent_directory}/data/test_dataset",
    )

    # 3. formatting dataset & save
    default_dataset = load_from_disk(f"{current_directory}/data/train_dataset")
    default_train_datast = default_dataset["train"]
    default_validation_datast = default_dataset["validation"]

    train_dataset = Dataset.from_dict(
        {
            "id": [guid for guid in default_train_datast["id"]],
            "title": [title for title in default_train_datast["title"]],
            "context": [context for context in default_train_datast["context"]],
            "question": [question for question in default_train_datast["question"]],
            "answers": Dataset.from_dict(
                {
                    "text": [
                        answers["text"] for answers in default_train_datast["answers"]
                    ],
                    "answer_start": [
                        answers["answer_start"]
                        for answers in default_train_datast["answers"]
                    ],
                }
            ),
        },
        features=get_standard_features(),
    )

    validation_dataset = Dataset.from_dict(
        {
            "id": [guid for guid in default_validation_datast["id"]],
            "title": [title for title in default_validation_datast["title"]],
            "context": [context for context in default_validation_datast["context"]],
            "question": [
                question for question in default_validation_datast["question"]
            ],
            "answers": Dataset.from_dict(
                {
                    "text": [
                        answers["text"]
                        for answers in default_validation_datast["answers"]
                    ],
                    "answer_start": [
                        answers["answer_start"]
                        for answers in default_validation_datast["answers"]
                    ],
                }
            ),
        },
        features=get_standard_features(),
    )

    final_dataset = DatasetDict(
        {"train": train_dataset, "validation": validation_dataset}
    )
    final_dataset.save_to_disk(f"{parent_directory}/data/default")

    # 4. download했던 폴더 삭제
    os.remove(file_name)
    shutil.rmtree(f"{current_directory}/data/")


def squad_kor_v1():
    squad_kor_v1_dataset = load_dataset("squad_kor_v1")
    squad_kor_v1_train_dataset = squad_kor_v1_dataset["train"]
    squad_kor_v1_validation_dataset = squad_kor_v1_dataset["validation"]

    train_dataset = Dataset.from_dict(
        {
            "id": [guid for guid in squad_kor_v1_train_dataset["id"]],
            "title": [title for title in squad_kor_v1_train_dataset["title"]],
            "context": [context for context in squad_kor_v1_train_dataset["context"]],
            "question": [
                question for question in squad_kor_v1_train_dataset["question"]
            ],
            "answers": Dataset.from_dict(
                {
                    "text": [
                        answers["text"]
                        for answers in squad_kor_v1_train_dataset["answers"]
                    ],
                    "answer_start": [
                        answers["answer_start"]
                        for answers in squad_kor_v1_train_dataset["answers"]
                    ],
                }
            ),
        },
        features=get_standard_features(),
    )

    validation_dataset = Dataset.from_dict(
        {
            "id": [guid for guid in squad_kor_v1_validation_dataset["id"]],
            "title": [title for title in squad_kor_v1_validation_dataset["title"]],
            "context": [
                context for context in squad_kor_v1_validation_dataset["context"]
            ],
            "question": [
                question for question in squad_kor_v1_validation_dataset["question"]
            ],
            "answers": Dataset.from_dict(
                {
                    "text": [
                        answers["text"]
                        for answers in squad_kor_v1_validation_dataset["answers"]
                    ],
                    "answer_start": [
                        answers["answer_start"]
                        for answers in squad_kor_v1_validation_dataset["answers"]
                    ],
                }
            ),
        },
        features=get_standard_features(),
    )

    final_dataset = DatasetDict(
        {"train": train_dataset, "validation": validation_dataset}
    )

    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(f"{parent_directory}/data/"):
        os.makedirs(f"{parent_directory}/data/")
    final_dataset.save_to_disk(f"{parent_directory}/data/squad_kor_v1")


def klue_mrc():
    klue_mrc_dataset = load_dataset("klue", "mrc")
    klue_mrc_train_dataset = klue_mrc_dataset["train"]
    klue_mrc_validation_dataset = klue_mrc_dataset["validation"]

    """
    train_dataset = Dataset.from_dict({
        "id": klue_mrc_train_dataset["guid"],
        "title": klue_mrc_train_dataset["title"],
        "context": klue_mrc_train_dataset["context"],
        "question": klue_mrc_train_dataset["question"],
        "answers": klue_mrc_train_dataset["answers"],
   }, features=get_standard_features())

    validation_dataset = Dataset.from_dict({
        "id": klue_mrc_validation_dataset["guid"],
        "title": klue_mrc_validation_dataset["title"],
        "context": klue_mrc_validation_dataset["context"],
        "question": klue_mrc_validation_dataset["question"],
        "answers": klue_mrc_validation_dataset["answers"],
   }, features=get_standard_features())
   """

    train_dataset = Dataset.from_dict(
        {
            "id": [guid for guid in klue_mrc_train_dataset["guid"]],
            "title": [title for title in klue_mrc_train_dataset["title"]],
            "context": [context for context in klue_mrc_train_dataset["context"]],
            "question": [question for question in klue_mrc_train_dataset["question"]],
            "answers": Dataset.from_dict(
                {
                    "text": [
                        answers["text"] for answers in klue_mrc_train_dataset["answers"]
                    ],
                    "answer_start": [
                        answers["answer_start"]
                        for answers in klue_mrc_train_dataset["answers"]
                    ],
                }
            ),
        },
        features=get_standard_features(),
    )

    validation_dataset = Dataset.from_dict(
        {
            "id": [guid for guid in klue_mrc_validation_dataset["guid"]],
            "title": [title for title in klue_mrc_validation_dataset["title"]],
            "context": [context for context in klue_mrc_validation_dataset["context"]],
            "question": [
                question for question in klue_mrc_validation_dataset["question"]
            ],
            "answers": Dataset.from_dict(
                {
                    "text": [
                        answers["text"]
                        for answers in klue_mrc_validation_dataset["answers"]
                    ],
                    "answer_start": [
                        answers["answer_start"]
                        for answers in klue_mrc_validation_dataset["answers"]
                    ],
                }
            ),
        },
        features=get_standard_features(),
    )

    final_dataset = DatasetDict(
        {"train": train_dataset, "validation": validation_dataset}
    )

    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(f"{parent_directory}/data/"):
        os.makedirs(f"{parent_directory}/data/")
    final_dataset.save_to_disk(f"{parent_directory}/data/klue_mrc")


def sparse_retrieval_neg_sampling():
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sparse_checkpoint_file_name = "bm25-morphs_model=BM25Okapi_tokenizer=Kkma"

    sparse_checkpoint_path = os.path.join(
        parent_directory + f"/retrieval_checkpoints/{sparse_checkpoint_file_name}"
    )
    if not os.path.isfile(sparse_checkpoint_path):
        raise FileNotFoundError("bm25-morphs sparse checkpoint file doesn't exist.")

    neg_num = 10
    with open(sparse_checkpoint_path, "rb") as file:
        os.chdir(parent_directory)
        retrieval = pickle.load(file)
        os.chdir(parent_directory + "/utils")

    default_dataset = get_dataset_list(["default"])[0]
    train_dataset = default_dataset["train"]
    validation_dataset = default_dataset["validation"]

    train_min_len = 10
    _, _, retrieval_docs, _ = retrieval.search(train_dataset["question"], k=100)
    neg_sample_list = []
    for context, retrieval_doc in zip(train_dataset["context"], retrieval_docs):
        idx, cnt = 0, 0
        neg_sample = []
        while len(retrieval_doc) > idx:
            if (
                context.replace(" ", "").replace("\n", "").replace("\\n", "")
                != retrieval_doc[idx]
                .replace(" ", "")
                .replace("\n", "")
                .replace("\\n", "")
            ) and (
                retrieval_doc[idx].replace(" ", "").replace("\n", "").replace("\\n", "")
                not in [
                    ns.replace(" ", "").replace("\n", "").replace("\\n", "")
                    for ns in neg_sample
                ]
            ):
                neg_sample.append(retrieval_doc[idx])
                cnt += 1
                if cnt >= neg_num:
                    break
            idx += 1
        if train_min_len > cnt:
            train_min_len = cnt
        neg_sample_list.append(neg_sample)
    train_dataset = train_dataset.add_column("negative_sample", neg_sample_list)

    val_min_len = 10
    _, _, retrieval_docs, _ = retrieval.search(validation_dataset["question"], k=100)
    neg_sample_list = []
    for answers, retrieval_doc in zip(validation_dataset["answers"], retrieval_docs):
        idx, cnt = 0, 0
        neg_sample = []
        while len(retrieval_doc) > idx:
            if (
                context.replace(" ", "").replace("\n", "").replace("\\n", "")
                != retrieval_doc[idx]
                .replace(" ", "")
                .replace("\n", "")
                .replace("\\n", "")
            ) and (
                retrieval_doc[idx].replace(" ", "").replace("\n", "").replace("\\n", "")
                not in [
                    ns.replace(" ", "").replace("\n", "").replace("\\n", "")
                    for ns in neg_sample
                ]
            ):
                neg_sample.append(retrieval_doc[idx])
                cnt += 1
                if cnt >= neg_num:
                    break
            idx += 1
        if val_min_len > cnt:
            val_min_len = cnt
        neg_sample_list.append(neg_sample)
    validation_dataset = validation_dataset.add_column(
        "negative_sample", neg_sample_list
    )

    final_dataset = DatasetDict(
        {"train": train_dataset, "validation": validation_dataset}
    )

    if not os.path.exists(f"{parent_directory}/data/"):
        os.makedirs(f"{parent_directory}/data/")
    final_dataset.save_to_disk(f"{parent_directory}/data/sparse_retrieval_neg_sampling")
    print(
        f"minimum size of train neg sample: {train_min_len}, minimum size of val neg sample: {val_min_len}"
    )

    
# paraphrasing된 질문을 넣은 데이터셋을 저장
def paraphrased():
    # paraphrased 데이터셋을 반환
    def paraphrase(dataset, paraphrased_questions):
        paraphrased_data = []
        for example in dataset:
            if example["question"] in paraphrased_questions:
                paraphrased_example = copy.deepcopy(example)
                paraphrased_example["question"] = paraphrased_questions[
                    example["question"]
                ]
                paraphrased_example["id"] = example["id"] + "_paraphrased"
                paraphrased_data.append(paraphrased_example)

        if paraphrased_data:
            paraphrased_dataset = Dataset.from_list(
                paraphrased_data, features=get_standard_features()
            )

        return paraphrased_dataset

    default_dataset_path = os.getenv("DIR_PATH") + "/level2-mrc-nlp-11/data/default"
    default_dataset = load_from_disk(default_dataset_path)
    train_dataset = default_dataset["train"]
    validation_dataset = default_dataset["validation"]
    paraphrased_questions_path = (
        os.getenv("DIR_PATH") + "/level2-mrc-nlp-11/data/paraphrased_questions.json"
    )

    with open(paraphrased_questions_path, "r", encoding="utf-8") as f:
        paraphrased_questions = json.load(f)

    paraphrased_train_dataset = paraphrase(train_dataset, paraphrased_questions)
    paraphrased_validation_dataset = paraphrase(
        validation_dataset, paraphrased_questions
    )

    final_dataset = DatasetDict(
        {
            "train": paraphrased_train_dataset,
            "validation": paraphrased_validation_dataset,
        }
    )

    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(f"{parent_directory}/data/"):
        os.makedirs(f"{parent_directory}/data/")
    final_dataset.save_to_disk(f"{parent_directory}/data/paraphrased")


if __name__ == "__main__":
    default()
