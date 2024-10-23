import streamlit as st


def view_answer(example):
    st.subheader("정답", divider="gray")

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
    else:
        st.write("정답이 공개되지 않은 테스트 데이터입니다.")


def view_predictions(example, nbest_prediction):
    st.subheader("예측", divider="gray")

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
            st.markdown("<hr style='margin-'>", unsafe_allow_html=True)


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
