def test(example):
    example["context"] = example["context"].replace(" ", "")
    return example
