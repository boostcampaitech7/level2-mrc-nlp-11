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


# ETRI_MRC_v1 데이터셋 구조 동일하기
def etri_mrc():
    data_pass = (
        "/home/ppg/Desktop/AI Tech 7기/NLP/2. MRC/try/try0/20181101_ETRI_MRC_v1.json"
    )
    with open(data_pass, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    formatted_dataset_dict = {
        "title": [],
        "context": [],
        "question": [],
        "id": [],
        "answers": None,
    }

    answer_starts = []
    answer_texts = []

    for qa in dataset["data"]:
        title = qa["title"]
        for paragraph in qa["paragraphs"]:
            context = paragraph["context"]
            for question_set in paragraph["qas"]:
                id = f"etri_{question_set['id']}"  #
                question = question_set["question"]

                answer_text = [question_set["answers"][0]["text"]]
                answer_start = [question_set["answers"][0]["answer_start"]]

                answer_starts.append(answer_start)
                answer_texts.append(answer_text)

                formatted_dataset_dict["id"].append(id)
                formatted_dataset_dict["question"].append(question)
                formatted_dataset_dict["context"].append(context)
                formatted_dataset_dict["title"].append(title)

    formatted_dataset_dict["answers"] = Dataset.from_dict(
        {"text": answer_texts, "answer_start": answer_starts}
    )

    formatted_dataset = Dataset.from_dict(
        formatted_dataset_dict, features=get_standard_features()
    )

    split_dataset = formatted_dataset.train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]
    validation_dataset = split_dataset["test"]

    final_dataset = DatasetDict(
        {"train": train_dataset, "validation": validation_dataset}
    )

    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.exists(f"{parent_directory}/data/"):
        os.makedirs(f"{parent_directory}/data/")
    final_dataset.save_to_disk(f"{parent_directory}/data/etri_mrc")


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


# train의 context, question, answer의 길이 분포에 맞게 data를 생성해주는 함수
def filtered_outliers(target_name: str):
    """
    파라미터로 이상치를 제거하고 싶은 파일의 폴더 이름을 입력하세요.
    """
    parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Load the reference dataset (e.g., 'default' dataset)
    reference_data_path = f"{parent_directory}/data/default"
    reference_dataset = load_from_disk(reference_data_path)
    reference_df = reference_dataset["train"].to_pandas()

    # Compute lengths in the reference dataset
    reference_lengths = {
        "context": reference_df["context"].str.len(),
        "question": reference_df["question"].str.len(),
        "answers_text": reference_df["answers"].apply(
            lambda x: (
                len(x["text"][0])
                if isinstance(x["text"], list) and len(x["text"]) > 0
                else 0
            )
        ),
    }

    # Compute IQR bounds
    def get_iqr_bounds(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = max(q1 - 1.5 * iqr, series.min())  # Ensure not less than min
        upper_bound = min(q3 + 1.5 * iqr, series.max())  # Ensure not more than max
        return lower_bound, upper_bound

    context_bounds = get_iqr_bounds(reference_lengths["context"])
    question_bounds = get_iqr_bounds(reference_lengths["question"])
    answers_text_bounds = get_iqr_bounds(reference_lengths["answers_text"])

    # Function to check if a value is within bounds
    def is_within_bounds(length, bounds):
        return bounds[0] <= length <= bounds[1]

    # Function to filter outliers from a dataset
    def filter_outliers(dataset, context_bounds, question_bounds, answers_text_bounds):
        df = dataset.to_pandas()

        # Calculate lengths
        df["context_len"] = df["context"].str.len()
        df["question_len"] = df["question"].str.len()
        df["answers_text_len"] = df["answers"].apply(
            lambda x: (
                len(x["text"][0])
                if isinstance(x["text"], list) and len(x["text"]) > 0
                else 0
            )
        )

        # Determine non-outliers
        mask = (
            df["context_len"].apply(lambda x: is_within_bounds(x, context_bounds))
            & df["question_len"].apply(lambda x: is_within_bounds(x, question_bounds))
            & df["answers_text_len"].apply(
                lambda x: is_within_bounds(x, answers_text_bounds)
            )
        )

        # Create filtered dataset
        filtered_df = df[mask].drop(
            columns=["context_len", "question_len", "answers_text_len"]
        )
        return Dataset.from_pandas(filtered_df, preserve_index=False)

    # Load the datasets you want to filter (e.g., 'paraphrased' dataset)
    dataset_to_filter_path = f"{parent_directory}/data/{target_name}"
    if not os.path.exists(dataset_to_filter_path):
        print(
            f"Dataset not found at {dataset_to_filter_path}. Please generate it first."
        )
        return

    dataset_to_filter = load_from_disk(dataset_to_filter_path)
    train_dataset = dataset_to_filter["train"]
    validation_dataset = dataset_to_filter["validation"]

    # Filter datasets
    filtered_train_dataset = filter_outliers(
        train_dataset, context_bounds, question_bounds, answers_text_bounds
    )
    filtered_validation_dataset = filter_outliers(
        validation_dataset, context_bounds, question_bounds, answers_text_bounds
    )

    # Save the filtered datasets
    final_dataset = DatasetDict(
        {"train": filtered_train_dataset, "validation": filtered_validation_dataset}
    )
    output_dataset_path = f"{parent_directory}/data/{target_name}_filtered"
    os.makedirs(output_dataset_path, exist_ok=True)
    final_dataset.save_to_disk(output_dataset_path)

    print(f"Filtered dataset saved to {output_dataset_path}")


if __name__ == "__main__":
    # default()
    filtered_outliers("aug_prev")
