import pandas as pd
import ast
from datasets import Dataset, Features, Sequence, Value

input_file_path = "/AI-projects/ODQA-project/data/aug_prev/train/aug_prev_train_update_id.csv"
output_file_path = (
    "/AI-projects/ODQA-project/data/aug_prev/train"
)
# Step 1: Read the CSV file
df = pd.read_csv(input_file_path)

# Step 2: Parse the 'answers' column
df["answers"] = df["answers"].apply(ast.literal_eval)

# Step 3: Ensure the 'answers' column has the correct structure
# This step ensures that 'answers' has a list of dictionaries with the proper structure
df["answers"] = df["answers"].apply(lambda x: {
    'answer_start': x['answer_start'],  # 리스트 형태 유지
    'text': x['text']
})

# Debug: Print the first few rows of the 'answers' column
print(df["answers"].head())

# Step 4: Define the features schema
features = Features(
    {
        "id": Value("string"),
        "title": Value("string"),
        "context": Value("string"),
        "question": Value("string"),
        "answers": {
            "answer_start": Sequence(Value("int64")), # Ensure int64 is used
            "text": Sequence(Value("string")),
        },
         "document_id": Value("int64"),               # document_id 추가
        "__index_level_0__": Value("int64")         # __index_level_0__ 추가
    }
)


# Step 5: Create the Dataset
try:
    dataset = Dataset.from_pandas(df, features=features)
except ValueError as e:
    print(f"Error: {e}")
    print(f"DataFrame structure: {df.dtypes}")
    print(df.head())

# Step 6: Save the Dataset to disk
if 'dataset' in locals():  # Check if dataset is defined
    dataset.save_to_disk(output_file_path)
else:
    print("Dataset could not be created due to earlier errors.")