from module.retrieval import SubwordBm25Retrieval, MorphsBm25Retrieval, TfIdfRetrieval
from utils.data_template import get_dataset_list
from datasets import concatenate_datasets
import hydra


@hydra.main(config_path="./config", config_name="retrieval", version_base=None)
def main(config):

    # 1. train retrieval by wiki docs
    # retrieval = SubwordBm25Retrieval(config.bm25.subword) # 1. subword base bm25
    # retrieval = MorphsBm25Retrieval(config.bm25.morphs) # 2. morphs base bm25
    retrieval = TfIdfRetrieval(config.tfidf)  # 3. subword base tf-idf
    retrieval.fit()
    retrieval.save()

    mode = "validation"
    top_k = 1

    # 2. evaluate model
    # 2.1. load eval examples
    dataset_list = get_dataset_list(config.data.dataset_name)
    if mode == "train":
        examples = concatenate_datasets([ds["train"] for ds in dataset_list])
    elif mode == "validation":
        examples = concatenate_datasets([ds["validation"] for ds in dataset_list])

    # 2.2. predict docs
    _, _, docs = retrieval.search(examples["question"], k=top_k)

    # 2.3. calculate accuracy
    cnt = 0
    for eval_example, doc in zip(examples, docs):
        if eval_example["context"].replace(" ", "").replace("\n", "").replace(
            "\\n", ""
        ) in [d.replace(" ", "").replace("\n", "").replace("\\n", "") for d in doc]:
            cnt += 1
        print(cnt)
    print(f"mode: {mode}, total: {len(examples)}, correct: {cnt}")


if __name__ == "__main__":
    main()
