import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module.data import *
import hydra
import json


def save_tokenized_samples(tokenizer, dataset, output_dir):

    tokenized_context = {
        dataset[i]["example_id"]: tokenizer.convert_ids_to_tokens(
            dataset[i]["input_ids"]
        )
        for i in range(len(dataset))
    }

    token_file = os.path.join(output_dir, "tokenized_samples.json")

    with open(token_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(tokenized_context, indent=4, ensure_ascii=False) + "\n")


# train.py에서 config.tokenizer.save = False로 해놔서 토큰 저장 못 했으면 이 파일을 직접 실행!
@hydra.main(config_path="..", config_name="config", version_base=None)
def main(config):
    data_module = MrcDataModule(config)
    data_module.setup()

    save_tokenized_samples(
        data_module.tokenizer, data_module.eval_dataset, config.train.output_dir
    )


if __name__ == "__main__":
    main()
