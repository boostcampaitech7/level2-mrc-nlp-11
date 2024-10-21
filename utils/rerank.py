import numpy as np


class RetrievalReranker:

    def __init__(self, sparse_retrieval, dense_retrieval):
        self.sparse_retrieval = sparse_retrieval
        self.dense_retrieval = dense_retrieval
        self.sparse_retrieval_contexts_idx_key_pair = {
            v: k for k, v in self.sparse_retrieval.contexts_key_idx_pair.items()
        }
        self.dense_retrieval_contexts_key_idx_pair = (
            self.dense_retrieval.contexts_key_idx_pair
        )

    def search(self, question, rerank_k=100, final_k=10):
        docs_score, docs_idx, docs, titles = self.sparse_retrieval.search(
            question, k=rerank_k
        )
        docs_key = [
            self.sparse_retrieval_contexts_idx_key_pair[doc_idx]
            for doc_idx in docs_idx[0]
        ]
        sparse_docs_key_score_pair = {
            doc_key: doc_score for doc_key, doc_score in zip(docs_key, docs_score[0])
        }
        key_doc_pair = {doc_key: doc for doc_key, doc in zip(docs_key, docs[0])}
        sparse_doc_score_max = np.max(docs_score)

        dense_retrieval_docs_idx = [
            self.dense_retrieval_contexts_key_idx_pair[key] for key in docs_key
        ]
        _, _, _, sim_score = self.dense_retrieval.search(
            question, k=rerank_k, return_sim_score=True
        )
        dense_docs_key_score_pair = {
            doc_key: doc_score
            for doc_key, doc_score in zip(docs_key, sim_score[dense_retrieval_docs_idx])
        }
        dense_doc_score_max = np.max(sim_score)

        final_docs_key_score_pair = {
            doc_key: sparse_docs_key_score_pair[doc_key] / sparse_doc_score_max
            + dense_docs_key_score_pair[doc_key] / dense_doc_score_max
            for doc_key in docs_key
        }
        final_docs_key_score_pair = sorted(
            final_docs_key_score_pair.items(), key=lambda item: -item[1]
        )[:final_k]
        final_docs = [key_doc_pair[key] for (key, score) in final_docs_key_score_pair]

        return None, None, final_docs
