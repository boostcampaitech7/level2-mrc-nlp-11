import streamlit as st
from datasets import Dataset
import copy


def view_answer(data_module, example):
    st.header("정답", divider="blue")

    if "answers" in example:
        context = example["context"]
        answer_start = example["answers"]["answer_start"][0]
        answer_end = answer_start + len(example["answers"]["text"][0])
        highlighted_context = (
            context[:answer_start]
            + f"<span style='background-color: #80ffdb; border-radius:10px; padding: 2px;'>{context[answer_start:answer_end]}</span>"
            + context[answer_end:]
        )

        st.markdown(
            f"""
                    <div style='background-color: #f7f7ff; border-radius: 10px; padding: 20px;'>
                    <div style='font-weight: bold; font-size: 20px; margin-bottom: 20px;'>문서 제목: {example['title']}</div>
                    {highlighted_context}
                    </div>
                    """,
            unsafe_allow_html=True,
        )

        st.subheader("Wanna see tokenized example?", divider="gray")
        view_tokenized_example(data_module, example, f"answer-{example['id']}")
    else:
        st.write("정답이 공개되지 않은 테스트 데이터입니다.")


def view_predictions(data_module, example, nbest_prediction):
    st.header("예측", divider="blue")

    if nbest_prediction is None:
        st.write("예측을 하지 않는 학습 데이터입니다.")
        return
    # final prediction(best-1)
    prediction = nbest_prediction[0]
    prediction_text = prediction["text"]
    prediction_start = prediction["start"]
    prediction_end = prediction_start + len(prediction_text)
    context = prediction["context"]
    if "answers" in example:
        span_color = (
            "#80ffdb" if example["answers"]["text"][0] == prediction_text else "#ffb3c6"
        )
    else:
        span_color = "#b8b8ff"

    highlighted_context = (
        context[:prediction_start]
        + f"<span style='background-color: {span_color}; border-radius:10px; padding: 2px;'>{context[prediction_start:prediction_end]}</span>"
        + context[prediction_end:]
    )

    st.markdown(
        f"""
                <div style='background-color: #f7f7ff; border-radius: 10px; padding: 20px; margin-bottom: 40px;'>
                {highlighted_context}
                </div>
                """,
        unsafe_allow_html=True,
    )

    example_for_tokenize = copy.deepcopy(example)
    example_for_tokenize["context"] = prediction["context"]
    st.subheader("Wanna see tokenized example?", divider="gray")
    view_tokenized_example(data_module, example_for_tokenize, f"pred-{example['id']}")

    with st.expander("nbest prediction 보기"):
        for i in range(1, len(nbest_prediction)):
            prediction = nbest_prediction[i]
            prediction_text = prediction["text"]
            prediction_start = prediction["start"]
            prediction_end = prediction_start + len(prediction_text)
            context = prediction["context"]
            if "answers" in example:
                span_color = (
                    "#80ffdb"
                    if example["answers"]["text"][0] == prediction_text
                    else "#ffb3c6"
                )
            else:
                span_color = "#b8b8ff"

            highlighted_context = (
                context[:prediction_start]
                + f"<span style='background-color: {span_color}; border-radius:10px; padding: 2px;'>{context[prediction_start:prediction_end]}</span>"
                + context[prediction_end:]
            )

            st.markdown(
                f"""
                        <div style='padding: 0px 20px;'>
                        {highlighted_context}
                        """,
                unsafe_allow_html=True,
            )
            example_for_tokenize = copy.deepcopy(example)
            example_for_tokenize["context"] = prediction["context"]
            view_tokenized_example(
                data_module, example_for_tokenize, f"nbest{i}-{example['id']}"
            )
            st.markdown("<hr>", unsafe_allow_html=True)


def view_documents(documents):
    col1, col2 = st.columns(2)

    with col1:
        for i in range(0, len(documents), 2):
            document = documents[i]
            with st.expander(f'{document["document_id"]}: {document["title"]}'):
                st.markdown(document["text"])

    with col2:
        for i in range(1, len(documents), 2):
            document = documents[i]
            with st.expander(f'{document["document_id"]}: {document["title"]}'):
                st.markdown(document["text"])


def view_tokenized_example(data_module, example, key):
    tokenize_button = st.button("Tokenize🤖", key=key)

    if tokenize_button:
        examples = Dataset.from_list([example])
        data = data_module.get_dataset(
            examples, data_module.prepare_validation_features
        )
        tokenized_examples = []
        for i in range(len(data[0]["input_ids"])):
            # Tokenizing
            tokenized_examples.append(
                data_module.tokenizer.convert_ids_to_tokens(data[0]["input_ids"][i])
            )

        for tokenized_example in tokenized_examples:
            colored_tokens = []
            for token in tokenized_example:
                # 스페셜 토큰 처리
                if token in ["[SEP]", "[CLS]"]:
                    colored_tokens.append(
                        f"<div style='display:inline-block; font-size:14px; background-color:#c5cff6; border:1px solid #ddd; border-radius:5px; padding:1px 5px; margin:2px;'>{token}</div>"
                    )
                # 패드 토큰
                elif token == "[PAD]":
                    colored_tokens.append(
                        f"<div style='display:inline-block; font-size:14px; color: #ddd; border:1px solid #ddd; border-radius:5px; padding:1px 5px; margin:2px;'>{token}</div>"
                    )
                else:
                    colored_tokens.append(
                        f"<div style='display:inline-block; font-size:14px; border:1px solid #ddd; border-radius:5px; padding:1px 5px; margin:2px;'>{token}</div>"
                    )

            st.markdown(
                f"""
                        <div style='background-color: #f7f7ff; border-radius: 10px; padding: 20px; margin-bottom: 20px;'>
                        {" ".join(colored_tokens)}
                        </div>
                        """,
                unsafe_allow_html=True,
            )
