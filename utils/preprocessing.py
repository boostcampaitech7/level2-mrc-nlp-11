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
from functools import wraps
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import joblib
import threading

class ModelPredictorSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ModelPredictorSingleton, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path, tokenizer_path, label_encoder_path, device=None):
        if self._initialized:
            return
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.label_encoder_path = label_encoder_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self._initialized = True

    def load_resources(self):
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        if self.label_encoder is None:
            self.label_encoder = joblib.load(self.label_encoder_path)

    def predict(self, text):
        self.load_resources()
        # 예측 로직은 이전과 동일합니다.
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).cpu().item()
            category = self.label_encoder.inverse_transform([predicted_class])[0]
        return category

def categorize_question(example):
    # Singleton 인스턴스 생성 (필요할 때 한 번만 생성)
    model_predictor = ModelPredictorSingleton(
        model_path='/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/question_classification/bert-base-multilingual-cased',
        tokenizer_path='/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/question_classification/bert-base-multilingual-cased',
        label_encoder_path='/data/ephemeral/home/sangyeop/level2-mrc-nlp-11/question_classification/bert-base-multilingual-cased/label_encoder.joblib'
    )
    category = model_predictor.predict(example['question'])
    example['question'] = example['question'] + '<' + category + '>'
    return example

def test(example):
    example["context"] = example["context"].replace("\n", " ")
    return example


def title_context_merge(example):
    # 제목을 강조하기 위해 특정 토큰을 추가합니다.
    title = f"<TITLE> {example['title']} <TITLE_END> "
    for idx in range(len(example["answers"]["answer_start"])):
        example["answers"]["answer_start"][idx] = example["answers"]["answer_start"][
            idx
        ] + len(title)
    example["context"] = f"{title}{example['context']}"
    return example
