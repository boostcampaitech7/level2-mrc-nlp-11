import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.data import MrcDataModule
from module.retrieval import TfIdfRetrieval
import hydra
import json
import pandas as pd


def save_tokenized_samples(tokenizer, dataset, output_dir, prefix):

    tokenized_context = {
        dataset[i]["example_id"]: tokenizer.convert_ids_to_tokens(
            dataset[i]["input_ids"]
        )
        for i in range(len(dataset))
    }

    token_file = os.path.join(output_dir, f"{prefix}_tokenized_samples.json")

    with open(token_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(tokenized_context, indent=4, ensure_ascii=False) + "\n")


@hydra.main(config_path="../config", config_name="combine", version_base=None)
def main(config):
    train_token_file = os.path.join(
        config.mrc.train.output_dir, f"train_tokenized_samples.json"
    )
    eval_token_file = os.path.join(
        config.mrc.train.output_dir, f"eval_tokenized_samples.json"
    )
    if not (os.path.exists(train_token_file) and os.path.exists(eval_token_file)):
        print("################################")
        print("train/eval dataset의 tokenized_sample이 없습니다. 지금 생성합니다.")
        print("################################")

        data_module = MrcDataModule(config.mrc)
        data_module.setup()

        # eval_tokenized_samples
        save_tokenized_samples(
            data_module.tokenizer,
            data_module.eval_dataset,
            config.mrc.train.output_dir,
            "eval",
        )
        # train_tokenized_samples
        save_tokenized_samples(
            data_module.tokenizer,
            data_module.train_dataset,
            config.mrc.train.output_dir,
            "train",
        )


if __name__ == "__main__":
    main()
