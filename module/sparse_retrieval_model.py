import json, os, pickle
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
import konlpy.tag as morphs_analyzer

from utils import rank_bm25


class CombineBm25Retrieval:
    def __init__(self, config):
        self.config = config
        assert self.config.morphs.analyzer_name == "Kkma"
        self.tag_set = self.get_tag_set()
        self.analyzer = getattr(morphs_analyzer, self.config.morphs.analyzer_name)()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.subword.tokenizer_name
        )
        self.morphs_bm25 = None
        self.subword_bm25 = None
        (
            self.titles,
            self.contexts,
            self.tokenized_contexts,
            self.contexts_key_idx_pair,
        ) = self.prepare_data()

    def prepare_data(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.morphs.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text_key_pair = {
            v["text"]: (k, v[self.config.morphs.tokenized_column])
            for k, v in data.items()
        }
        titles, contexts, tokenized_contexts, contexts_key_idx_pair = [], [], [], {}
        for idx, (text, (k, tokenized_text)) in enumerate(text_key_pair.items()):
            titles.append(data[k]["title"])
            contexts.append(text)
            tokenized_contexts.append(tokenized_text)
            contexts_key_idx_pair[k] = idx
        return titles, contexts, tokenized_contexts, contexts_key_idx_pair

    def tokenize(self, context):
        tokenized_context = []
        for token, pos in self.analyzer.pos(context):
            if pos in self.tag_set:
                continue
            tokenized_context.append(token)

        return tokenized_context, self.tokenizer.tokenize(context)

    def get_tag_set(self):
        if self.config.morphs.analyzer_name == "Kkma":
            josa_tag_list = [
                "JKS",
                "JKC",
                "JKG",
                "JKO",
                "JKM",
                "JKI",
                "JKQ",
                "JC",
                "JX",
            ]
            suffix_list = [
                "EPH",
                "EPT",
                "EPP",
                "EFN",
                "EFQ",
                "EFO",
                "EFA",
                "EFI",
                "EFR",
                "ECE",
                "ECS",
                "ECD",
                "ETN",
                "ETD",
                "XSN",
                "XSV",
                "XSA",
            ]
            etc_list = ["IC", "SF", "SE", "SS", "SP", "SO"]
            return set(josa_tag_list + suffix_list + etc_list)
        return None

    def fit(self):
        self.morphs_bm25 = getattr(rank_bm25, self.config.morphs.model)(
            self.tokenized_contexts
        )
        del self.tokenized_contexts
        tokenized_contexts = []

        for context in self.contexts:
            tokenized_contexts.append(self.tokenizer.tokenize(context))
        self.subword_bm25 = getattr(rank_bm25, self.config.subword.model)(
            tokenized_contexts
        )

    def save(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bm25_save_dir = f"{parent_directory}/retrieval_checkpoints/"
        bm25_model_path = (
            bm25_save_dir
            + f"combine-bm25_model={self.config.morphs.model}_analyzer={self.config.morphs.analyzer_name}_tokenizer={self.config.subword.tokenizer_name}".replace(
                "/", "-"
            )
        )
        if not os.path.isdir(bm25_save_dir):
            os.makedirs(bm25_save_dir)
        with open(bm25_model_path, "wb") as file:
            self.analyzer = None
            pickle.dump(self, file)

    def search(self, query_list, k=1, return_query_score=False):
        if not self.analyzer:
            self.analyzer = getattr(morphs_analyzer, self.config.morphs.analyzer_name)()

        if isinstance(query_list, str):
            query_list = [query_list]

        tokenized_query_list = []
        for query in query_list:
            morphs_token, subword_token = self.tokenize(query)
            tokenized_query_list.append((morphs_token, subword_token))

        titles, docs_score, docs_idx, docs, query_score_list = [], [], [], [], []
        for morphs_token, subword_token in tokenized_query_list:
            morphs_doc_score, morphs_query_score = self.morphs_bm25.get_scores(
                morphs_token
            )
            subword_doc_score, subword_query_score = self.subword_bm25.get_scores(
                subword_token
            )
            doc_score = 0.3 * (morphs_doc_score / max(morphs_doc_score)) + 0.7 * (
                subword_doc_score / max(subword_doc_score)
            )

            sorted_idx = np.argsort(doc_score)[::-1]
            docs_score.append(doc_score[sorted_idx][:k])
            docs_idx.append(sorted_idx[:k])
            docs.append([self.contexts[idx] for idx in sorted_idx[:k]])
            titles.append([self.titles[idx] for idx in sorted_idx[:k]])
            if return_query_score:
                query_score_list.append({**morphs_query_score, **subword_query_score})

        if not return_query_score:
            return docs_score, docs_idx, docs, titles

        return docs_score, docs_idx, docs, titles, query_score_list


class MorphsBm25Retrieval:
    def __init__(self, config):
        self.config = config
        assert self.config.analyzer_name == "Kkma"
        self.analyzer = getattr(morphs_analyzer, self.config.analyzer_name)()
        self.tag_set = self.get_tag_set()
        self.bm25 = None
        (
            self.titles,
            self.contexts,
            self.tokenized_contexts,
            self.contexts_key_idx_pair,
        ) = self.prepare_data()

    def prepare_data(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text_key_pair = {
            v["text"]: (k, v[self.config.tokenized_column]) for k, v in data.items()
        }
        titles, contexts, tokenized_contexts, contexts_key_idx_pair = [], [], [], {}
        for idx, (text, (k, tokenized_text)) in enumerate(text_key_pair.items()):
            titles.append(data[k]["title"])
            if self.config.add_title:
                tokenized_text = self.tokenize(data[k]["title"]) + " " + tokenized_text
                text = data[k]["title"] + " " + text
            contexts.append(text)
            tokenized_contexts.append(tokenized_text)
            contexts_key_idx_pair[k] = idx
        return titles, contexts, tokenized_contexts, contexts_key_idx_pair

    def tokenize(self, context):
        tokenized_context = []
        for token, pos in self.analyzer.pos(context):
            if pos in self.tag_set:
                continue
            tokenized_context.append(token)
        return tokenized_context

    def get_tag_set(self):
        if self.config.analyzer_name == "Kkma":
            josa_tag_list = [
                "JKS",
                "JKC",
                "JKG",
                "JKO",
                "JKM",
                "JKI",
                "JKQ",
                "JC",
                "JX",
            ]
            suffix_list = [
                "EPH",
                "EPT",
                "EPP",
                "EFN",
                "EFQ",
                "EFO",
                "EFA",
                "EFI",
                "EFR",
                "ECE",
                "ECS",
                "ECD",
                "ETN",
                "ETD",
                "XSN",
                "XSV",
                "XSA",
            ]
            etc_list = ["IC", "SF", "SE", "SS", "SP", "SO"]
            return set(josa_tag_list + suffix_list + etc_list)
        return None

    def fit(self):
        self.bm25 = getattr(rank_bm25, self.config.model)(self.tokenized_contexts)
        del self.tokenized_contexts

    def save(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bm25_save_dir = f"{parent_directory}/retrieval_checkpoints/"
        bm25_model_path = (
            bm25_save_dir
            + f"bm25-morphs_model={self.config.model}_tokenizer={self.config.analyzer_name}".replace(
                "/", "-"
            )
        )
        if not os.path.isdir(bm25_save_dir):
            os.makedirs(bm25_save_dir)
        with open(bm25_model_path, "wb") as file:
            self.analyzer = None
            pickle.dump(self, file)

    def search(self, query_list, k=1, return_query_score=False):
        if not self.analyzer:
            self.analyzer = getattr(morphs_analyzer, self.config.analyzer_name)()

        if isinstance(query_list, str):
            query_list = [query_list]

        tokenized_query_list = []
        for query in query_list:
            tokenized_query_list.append(self.tokenize(query))

        titles, docs_score, docs_idx, docs, query_score_list = [], [], [], [], []
        for tokenized_query in tokenized_query_list:
            doc_score, query_score = self.bm25.get_scores(tokenized_query)
            sorted_idx = np.argsort(doc_score)[::-1]

            docs_score.append(doc_score[sorted_idx][:k])
            docs_idx.append(sorted_idx[:k])
            docs.append([self.contexts[idx] for idx in sorted_idx[:k]])
            titles.append([self.titles[idx] for idx in sorted_idx[:k]])
            if return_query_score:
                query_score_list.append(query_score)

        if not return_query_score:
            return docs_score, docs_idx, docs, titles

        return docs_score, docs_idx, docs, titles, query_score_list


class SubwordBm25Retrieval:

    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self.bm25 = None
        self.titles, self.contexts, self.contexts_key_idx_pair = self.prepare_data()

    def prepare_data(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text_key_pair = {v["text"]: k for k, v in data.items()}
        titles, contexts, contexts_key_idx_pair = [], [], {}
        for idx, (text, k) in enumerate(text_key_pair.items()):
            titles.append(data[k]["title"])
            if self.config.add_title:
                text = data[k]["title"] + " " + text
            contexts.append(text)
            contexts_key_idx_pair[k] = idx
        return titles, contexts, contexts_key_idx_pair

    def fit(self):
        tokenized_contexts = []
        for context in self.contexts:
            tokenized_contexts.append(self.tokenizer.tokenize(context))
        self.bm25 = getattr(rank_bm25, self.config.model)(tokenized_contexts)

    def save(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bm25_save_dir = f"{parent_directory}/retrieval_checkpoints/"
        bm25_model_path = (
            bm25_save_dir
            + f"bm25-subword_model={self.config.model}_tokenizer={self.config.tokenizer_name}".replace(
                "/", "-"
            )
        )
        if not os.path.isdir(bm25_save_dir):
            os.makedirs(bm25_save_dir)
        with open(bm25_model_path, "wb") as file:
            pickle.dump(self, file)

    def search(self, query_list, k=1, return_query_score=False):
        if isinstance(query_list, str):
            query_list = [query_list]

        tokenized_query_list = []
        for query in query_list:
            tokenized_query_list.append(self.tokenizer.tokenize(query))

        titles, docs_score, docs_idx, docs, query_score_list = [], [], [], [], []
        for tokenized_query in tokenized_query_list:
            doc_score, query_score = self.bm25.get_scores(tokenized_query)
            sorted_idx = np.argsort(doc_score)[::-1]

            docs_score.append(doc_score[sorted_idx][:k])
            docs_idx.append(sorted_idx[:k])
            docs.append([self.contexts[idx] for idx in sorted_idx[:k]])
            titles.append([self.titles[idx] for idx in sorted_idx[:k]])
            if return_query_score:
                query_score_list.append(query_score)

        if not return_query_score:
            return docs_score, docs_idx, docs, titles

        return docs_score, docs_idx, docs, titles, query_score_list


class TfIdfRetrieval:

    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.tokenizer.tokenize,
            ngram_range=tuple(self.config.ngram),
            max_features=50000,
        )
        self.sparse_embedding_matrix = None
        self.titles, self.contexts, self.contexts_key_idx_pair = self.prepare_data()

    def prepare_data(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        context_data_path = f"{parent_directory}/{self.config.data_path}"

        with open(context_data_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        text_key_pair = {v["text"]: k for k, v in data.items()}
        titles, contexts, contexts_key_idx_pair = [], [], {}
        for idx, (text, k) in enumerate(text_key_pair.items()):
            titles.append(data[k]["title"])
            if self.config.add_title:
                text = data[k]["title"] + " " + text
            contexts.append(text)
            contexts_key_idx_pair[k] = idx
        return titles, np.array(contexts), contexts_key_idx_pair

    def fit(self):
        self.sparse_embedding_matrix = self.vectorizer.fit_transform(self.contexts)

    def save(self):
        parent_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tfidf_save_dir = f"{parent_directory}/retrieval_checkpoints/"
        tfidf_model_path = (
            tfidf_save_dir
            + f"tf-idf_tokenizer={self.config.tokenizer_name}_ngram={self.config.ngram}".replace(
                "/", "-"
            )
        )
        if not os.path.isdir(tfidf_save_dir):
            os.makedirs(tfidf_save_dir)
        with open(tfidf_model_path, "wb") as file:
            pickle.dump(self, file)

    def search(self, query_list, k=1, return_query_score=False):
        if isinstance(query_list, str):
            query_list = [query_list]

        titles, docs_score, docs_idx, docs = [], [], [], []
        query_vector_list = self.vectorizer.transform(query_list)
        similarity = query_vector_list * self.sparse_embedding_matrix.T

        for i in range(len(query_list)):
            sorted_idx = np.argsort(similarity[i].data)[::-1]
            doc_idx = similarity[i].indices[sorted_idx][:k]
            doc_score = similarity[i].data[sorted_idx][:k]

            docs_score.append(doc_score)
            docs_idx.append(doc_idx)
            docs.append(list(self.contexts[doc_idx]))
            titles.append([self.titles[idx] for idx in doc_idx])

        if not return_query_score:
            return docs_score, docs_idx, docs, titles

        query_score_list = []
        vocab = sorted(self.vectorizer.vocabulary_.keys())
        sparse_embedding_matrix = self.sparse_embedding_matrix.toarray()

        for query_vector in query_vector_list:
            q_scores = defaultdict(lambda: np.zeros(sparse_embedding_matrix.shape[0]))
            for idx, data in zip(query_vector.indices, query_vector.data):
                q_scores[vocab[idx]] += data * sparse_embedding_matrix[:, idx]
            query_score_list.append(q_scores)

        return docs_score, docs_idx, docs, titles, query_score_list
