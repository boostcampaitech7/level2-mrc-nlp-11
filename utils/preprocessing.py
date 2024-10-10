def test(example):
    example["context"] = example["context"].replace("\n", " ")
    return example
