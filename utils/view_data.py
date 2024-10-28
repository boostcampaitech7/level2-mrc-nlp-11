from data_template import get_dataset_list

dataset_list = get_dataset_list(["aug_prev_filtered"])
print(dataset_list[0]["train"][0])
breakpoint()
