import collections, json, logging, os
from typing import Optional, Tuple
from tqdm.auto import tqdm
import numpy as np
import torch
import pytorch_lightning as pl
from transformers import AutoModelForQuestionAnswering, EvalPrediction
from evaluate import load
from utils.common import init_obj
import module.metric as module_metric
from konlpy.tag import Kkma


logger = logging.getLogger(__name__)


class MrcLightningModule(pl.LightningModule):
    def __init__(
        self,
        config,
        eval_dataset=None,
        test_dataset=None,
        eval_examples=None,
        test_examples=None,
        inference_mode=None,
    ):
        super().__init__()
        self.kkma = Kkma()
        self.save_hyperparameters()
        self.config = config
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.config.model.plm_name
        )
        # model의 임베딩 크기 수정
        if len(self.config.data.add_special_token) != 0:
            self.model.resize_token_embeddings(
                self.model.config.vocab_size + len(self.config.data.add_special_token)
            )

        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.eval_examples = eval_examples
        self.test_examples = test_examples
        self.step_outputs = {"start_logits": [], "end_logits": []}
        self.metric_list = {
            metric: {"method": load(metric), "wrapper": getattr(module_metric, metric)}
            for metric in self.config.metric
        }
        self.inference_mode = inference_mode

    def on_save_checkpoint(self, checkpoint):
        del checkpoint["hyper_parameters"]["eval_dataset"]
        del checkpoint["hyper_parameters"]["test_dataset"]
        del checkpoint["hyper_parameters"]["eval_examples"]
        del checkpoint["hyper_parameters"]["test_examples"]

    def on_load_checkpoint(self, checkpoint):
        self.eval_dataset = None
        self.test_dataset = None
        self.eval_examples = None
        self.test_examples = None

    def configure_optimizers(self):
        trainable_params = list(
            filter(lambda p: p.requires_grad, self.model.parameters())
        )

        optimizer_name = self.config.optimizer.name
        del self.config.optimizer.name
        optimizer = init_obj(
            optimizer_name, self.config.optimizer, torch.optim, trainable_params
        )
        return optimizer

    def training_step(self, batch):
        qa_output = self.model(**batch)
        self.log("step_train_loss", qa_output["loss"])
        return {"loss": qa_output["loss"]}

    def validation_step(self, batch):
        qa_output = self.model(**batch)
        self.step_outputs["start_logits"].extend(qa_output["start_logits"].cpu())
        self.step_outputs["end_logits"].extend(qa_output["end_logits"].cpu())

    def on_validation_epoch_end(self):
        eval_preds = self.post_processing_function(
            self.eval_examples,
            self.eval_dataset,
            (
                np.array(self.step_outputs["start_logits"]).squeeze(),
                np.array(self.step_outputs["end_logits"]).squeeze(),
            ),
        )
        self.step_outputs = {"start_logits": [], "end_logits": []}

        # compute metric
        for metric in self.metric_list.values():
            metric_result = metric["wrapper"](eval_preds, metric["method"])
            for k, v in metric_result.items():
                self.log(k, v)

    def test_step(self, batch):
        qa_output = self.model(**batch)
        self.step_outputs["start_logits"].extend(qa_output["start_logits"].cpu())
        self.step_outputs["end_logits"].extend(qa_output["end_logits"].cpu())

    def on_test_epoch_end(self):
        self.post_processing_function(
            self.test_examples,
            self.test_dataset,
            (
                np.array(self.step_outputs["start_logits"]).squeeze(),
                np.array(self.step_outputs["end_logits"]).squeeze(),
            ),
            "test",
        )
        self.step_outputs = {"start_logits": [], "end_logits": []}

    def predict(self, context, answer):
        pass

    def post_processing_function(self, examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        if self.inference_mode == "separate":
            predictions = self.postprocess_qa_predictions_separate_inference(
                examples=examples,
                features=features,
                predictions=predictions,
                n_best_size=self.config.data.n_best_size,
                max_answer_length=self.config.data.max_answer_length,
                output_dir=self.config.train.output_dir,
                prefix=stage,
            )
        else:
            predictions = self.postprocess_qa_predictions(
                examples=examples,
                features=features,
                predictions=predictions,
                n_best_size=self.config.data.n_best_size,
                max_answer_length=self.config.data.max_answer_length,
                output_dir=self.config.train.output_dir,
                prefix=stage,
            )
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        if not "answers" in examples.column_names:
            return formatted_predictions

        if self.inference_mode == "separate":
            references = []
            for example in examples:
                example_id = example["id"].split("_top")[0]
                for reference in references:
                    if reference["id"] == example_id:
                        break
                else:
                    references.append({"id": example_id, "answers": example["answers"]})
        else:
            references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    # 조사 제거 후처리 함수
    def remove_last_josa(self, answer):
        last_pos = self.kkma.pos(answer)[-1]
        if last_pos[1] in ["JKS", "JKC", "JKG", "JKO", "JKM", "JKI", "JKQ", "JC", "JX"]:
            position = answer.rfind(last_pos[0])
            if position + len(last_pos[0]) == len(answer):
                answer = answer[:position]
        return answer

    def postprocess_qa_predictions(
        self,
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        log_level: Optional[int] = logging.WARNING,
    ):
        """
        Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
        original contexts. This is the base postprocessing functions for models that only return start and end logits.

        Args:
            examples: The non-preprocessed dataset (see the main script for more information).
            features: The processed dataset (see the main script for more information).
            predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
                The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
                first dimension must match the number of elements of :obj:`features`.
            version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the underlying dataset contains examples with no answers.
            n_best_size (:obj:`int`, `optional`, defaults to 20):
                The total number of n-best predictions to generate when looking for an answer.
            max_answer_length (:obj:`int`, `optional`, defaults to 30):
                The maximum length of an answer that can be generated. This is needed because the start and end predictions
                are not conditioned on one another.
            null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
                The threshold used to select the null answer: if the best answer has a score that is less than the score of
                the null answer minus this threshold, the null answer is selected for this example (note that the score of
                the null answer for an example giving several features is the minimum of the scores for the null answer on
                each feature: all features must be aligned on the fact they `want` to predict a null answer).

                Only useful when :obj:`version_2_with_negative` is :obj:`True`.
            output_dir (:obj:`str`, `optional`):
                If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
                :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
                answers, are saved in `output_dir`.
            prefix (:obj:`str`, `optional`):
                If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
            log_level (:obj:`int`, `optional`, defaults to ``logging.WARNING``):
                ``logging`` log level (e.g., ``logging.WARNING``)
        """
        if len(predictions) != 2:
            raise ValueError(
                "`predictions` should be a tuple with two elements (start_logits, end_logits)."
            )
        all_start_logits, all_end_logits = predictions

        if len(predictions[0]) != len(features):
            raise ValueError(
                f"Got {len(predictions[0])} predictions and {len(features)} features."
            )

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        if version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        # Logging.
        logger.setLevel(log_level)
        logger.info(
            f"Post-processing {len(examples)} example predictions split into {len(features)} features."
        )

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_prediction = None
            prelim_predictions = []

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]
                # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
                # available in the current feature.
                token_is_max_context = features[feature_index].get(
                    "token_is_max_context", None
                )

                # Update minimum null prediction.
                feature_null_score = start_logits[0] + end_logits[0]
                if (
                    min_null_prediction is None
                    or min_null_prediction["score"] > feature_null_score
                ):
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
                end_indexes = np.argsort(end_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[end_index]) < 2
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue
                        # Don't consider answer that don't have the maximum context available (if such information is
                        # provided).
                        if (
                            token_is_max_context is not None
                            and not token_is_max_context.get(str(start_index), False)
                        ):
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (
                                    offset_mapping[start_index][0],
                                    offset_mapping[end_index][1],
                                ),
                                "score": start_logits[start_index]
                                + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
            if version_2_with_negative and min_null_prediction is not None:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # Only keep the best `n_best_size` predictions.
            predictions = sorted(
                prelim_predictions, key=lambda x: x["score"], reverse=True
            )[:n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score.
            if (
                version_2_with_negative
                and min_null_prediction is not None
                and not any(p["offsets"] == (0, 0) for p in predictions)
            ):
                predictions.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = self.remove_last_josa(context[offsets[0] : offsets[1]])
                pred["start"] = offsets[0]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""
            ):
                predictions.insert(
                    0,
                    {
                        "text": "empty",
                        "start_logit": 0.0,
                        "end_logit": 0.0,
                        "score": 0.0,
                        "start": 0,
                    },
                )

            # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
            # the LogSumExp trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction.
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = (
                    null_score
                    - best_non_null_pred["start_logit"]
                    - best_non_null_pred["end_logit"]
                )
                scores_diff_json[example["id"]] = float(
                    score_diff
                )  # To be JSON-serializable.
                if score_diff > null_score_diff_threshold:
                    all_predictions[example["id"]] = ""
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example["id"]] = [
                {
                    k: (
                        float(v)
                        if isinstance(v, (np.float16, np.float32, np.float64))
                        else v
                    )
                    for k, v in pred.items()
                }
                for pred in predictions
            ]

        # If we have an output_dir, let's save all those dicts.
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise EnvironmentError(f"{output_dir} is not a directory.")

            parent_directory = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

            run_name = f"{self.config.data.preproc_list[0]}_{self.config.data.dataset_name[0]}_bz={self.config.data.batch_size}_lr={self.config.optimizer.lr}"

            prediction_file = os.path.join(
                parent_directory,
                output_dir,
                (
                    f"{run_name}_predictions.json"
                    if prefix is None
                    else f"{run_name}_{prefix}_predictions.json"
                ),
            )
            nbest_file = os.path.join(
                parent_directory,
                output_dir,
                (
                    f"{run_name}_best_npredictions.json"
                    if prefix is None
                    else f"{run_name}_{prefix}_nbest_predictions.json"
                ),
            )
            if version_2_with_negative:
                null_odds_file = os.path.join(
                    parent_directory,
                    output_dir,
                    (
                        f"{run_name}_null_odds.json"
                        if prefix is None
                        else f"{run_name}_{prefix}_null_odds.json"
                    ),
                )

            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
                )
            logger.info(f"Saving nbest_preds to {nbest_file}.")
            with open(nbest_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
                )
            if version_2_with_negative:
                logger.info(f"Saving null_odds to {null_odds_file}.")
                with open(null_odds_file, "w", encoding="utf-8") as writer:
                    writer.write(
                        json.dumps(scores_diff_json, indent=4, ensure_ascii=False)
                        + "\n"
                    )

        return all_predictions

    def postprocess_qa_predictions_separate_inference(
        self,
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        log_level: Optional[int] = logging.WARNING,
    ):
        if len(predictions) != 2:
            raise ValueError(
                "`predictions` should be a tuple with two elements (start_logits, end_logits)."
            )
        all_start_logits, all_end_logits = predictions

        if len(predictions[0]) != len(features):
            raise ValueError(
                f"Got {len(predictions[0])} predictions and {len(features)} features."
            )

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        if version_2_with_negative:
            scores_diff_json = collections.OrderedDict()

        # Logging.
        logger.setLevel(log_level)
        logger.info(
            f"Post-processing {len(examples)} example predictions split into {len(features)} features."
        )

        # Let's loop over all the examples!
        prelim_predictions = {}
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_prediction = None
            # example_id_top1, example_id_top2, ...에서 원래 exampl_id 추출
            real_example_id = example["id"].split("_top")[0]
            prelim_predictions.setdefault(real_example_id, [])

            if example["doc_score"] < 0:
                doc_score = 1 - example["doc_score"]
            else:
                doc_score = example["doc_score"]

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]
                # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
                # available in the current feature.
                token_is_max_context = features[feature_index].get(
                    "token_is_max_context", None
                )

                # Update minimum null prediction.
                feature_null_score = start_logits[0] + end_logits[0] * doc_score

                if (
                    min_null_prediction is None
                    or min_null_prediction["score"] > feature_null_score
                ):
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                        "example_id": example_index,
                        "document_id": example["document_id"],
                    }

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
                end_indexes = np.argsort(end_logits)[
                    -1 : -n_best_size - 1 : -1
                ].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[end_index]) < 2
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue
                        # Don't consider answer that don't have the maximum context available (if such information is
                        # provided).
                        if (
                            token_is_max_context is not None
                            and not token_is_max_context.get(str(start_index), False)
                        ):
                            continue

                        prelim_predictions[real_example_id].append(
                            {
                                "offsets": (
                                    offset_mapping[start_index][0],
                                    offset_mapping[end_index][1],
                                ),
                                "score": (
                                    start_logits[start_index] + end_logits[end_index]
                                )
                                * doc_score,
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                                "example_idx": example_index,
                                "document_id": example["document_id"],
                            }
                        )
            if version_2_with_negative and min_null_prediction is not None:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

        for example_id, prelim_prediction_list in prelim_predictions.items():
            # Only keep the best `n_best_size` predictions.
            predictions = sorted(
                prelim_prediction_list, key=lambda x: x["score"], reverse=True
            )[:n_best_size]

            # # Add back the minimum null prediction if it was removed because of its low score.
            # if (
            #     version_2_with_negative
            #     and min_null_prediction is not None
            #     and not any(p["offsets"] == (0, 0) for p in predictions)
            # ):
            #     predictions.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.
            for pred in predictions:
                context = examples[pred["example_idx"]]["context"]
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]
                pred["start"] = offsets[0]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""
            ):
                predictions.insert(
                    0,
                    {
                        "text": "empty",
                        "start_logit": 0.0,
                        "end_logit": 0.0,
                        "score": 0.0,
                        "start": 0,
                        "example_idx": 0,
                        "document_id": 0,
                    },
                )

            # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
            # the LogSumExp trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not version_2_with_negative:
                all_predictions[example_id] = predictions[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction.
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = (
                    null_score
                    - best_non_null_pred["start_logit"]
                    - best_non_null_pred["end_logit"]
                )
                scores_diff_json[example_id] = float(
                    score_diff
                )  # To be JSON-serializable.
                if score_diff > null_score_diff_threshold:
                    all_predictions[example_id] = ""
                else:
                    all_predictions[example_id] = best_non_null_pred["text"]

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example_id] = [
                {
                    k: (
                        float(v)
                        if isinstance(v, (np.float16, np.float32, np.float64))
                        else v
                    )
                    for k, v in pred.items()
                }
                for pred in predictions
            ]

        # If we have an output_dir, let's save all those dicts.
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise EnvironmentError(f"{output_dir} is not a directory.")

            parent_directory = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )

            run_name = f"{self.config.data.preproc_list[0]}_{self.config.data.dataset_name[0]}_bz={self.config.data.batch_size}_lr={self.config.optimizer.lr}"

            prediction_file = os.path.join(
                parent_directory,
                output_dir,
                (
                    f"{run_name}_predictions.json"
                    if prefix is None
                    else f"{run_name}_{prefix}_predictions.json"
                ),
            )
            nbest_file = os.path.join(
                parent_directory,
                output_dir,
                (
                    f"{run_name}_best_npredictions.json"
                    if prefix is None
                    else f"{run_name}_{prefix}_nbest_predictions.json"
                ),
            )
            if version_2_with_negative:
                null_odds_file = os.path.join(
                    parent_directory,
                    output_dir,
                    (
                        f"{run_name}_null_odds.json"
                        if prefix is None
                        else f"{run_name}_{prefix}_null_odds.json"
                    ),
                )

            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
                )
            logger.info(f"Saving nbest_preds to {nbest_file}.")
            with open(nbest_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
                )
            if version_2_with_negative:
                logger.info(f"Saving null_odds to {null_odds_file}.")
                with open(null_odds_file, "w", encoding="utf-8") as writer:
                    writer.write(
                        json.dumps(scores_diff_json, indent=4, ensure_ascii=False)
                        + "\n"
                    )

        return all_predictions
