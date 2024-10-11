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

import re


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


def remove_markdown(example):
    # 마크다운 패턴 리스트를 함수 내부에 포함
    markdown_patterns = {
        "header": r"(?:^|\n)#{1,6}\s*",  # 헤더 (ex: #, ##, ###, ...) - 줄 시작 또는 줄바꿈 후 처리
        "list": r"^\s*[\*\-\+]\s+",  # 리스트 기호 제거 (ex: *, -, +) - 줄 시작에서 리스트 기호를 찾아 제거
        "bold": r"\*\*(.*?)\*\*",  # 굵게 (ex: **굵게**) 텍스트만 남김
        "italic": r"\*(.*?)\*",  # 기울임 (ex: *기울임*) 텍스트만 남김
        "code_block": r"```(.*?)```",  # 코드 블록 (ex: ```코드```)
        "inline_code": r"`(.*?)`",  # 인라인 코드 (ex: `코드`)
    }

    # 마크다운 문법에 해당하는 부분을 제거하거나 변환하는 과정
    for pattern_name, pattern in markdown_patterns.items():
        if pattern_name == "link":
            # 링크 패턴에 대해서는 링크만 남기고 텍스트는 제거
            example["context"] = re.sub(pattern, r"\2", example["context"])
        elif pattern_name == "bold" or pattern_name == "italic":
            # 굵게, 기울임은 텍스트만 남김
            example["context"] = re.sub(pattern, r"\1", example["context"])
        elif pattern_name == "code_block" or pattern_name == "inline_code":
            # 코드 블록과 인라인 코드는 그대로 남김
            example["context"] = re.sub(pattern, r"\1", example["context"])
        else:
            # 나머지 마크다운 문법에 해당하는 부분은 제거
            example["context"] = re.sub(
                pattern, "", example["context"], flags=re.MULTILINE
            )

    return example


def replace_markdown_with_tags(example):
    # 마크다운 패턴 리스트를 함수 내부에 포함
    markdown_patterns = {
        "header": (
            r"^\s*(#{1,6})\s*(.+)",
            r"<HEADER>\2",
        ),  # 헤더 (ex: #, ##, ###, ...) - <header>로 대체
        "list": (
            r"^\s*[\*\-\+]\s+(.+)",
            r"<LIST>\1",
        ),  # 리스트 (ex: *, -, +) - 리스트 기호 제거하고 <list>로 대체
        "bold": (
            r"\*\*(.*?)\*\*",
            r"<BOLD>\1<BOLD>",
        ),  # 굵게 (ex: **굵게**) - 굵게 텍스트 유지하고 <bold>
        "italic": (
            r"\*(.*?)\*",
            r"<ITALIC>\1<ITALIC>",
        ),  # 기울임 (ex: *기울임*) - 기울임 텍스트 유지하고 <italic>
        "code_block": (
            r"```(.*?)```",
            r"<CODE_BLOCK>\1<CODE_BLOCK>",
        ),  # 코드 블록 (ex: ```코드```) - 코드 블록 유지하고 <code_block>
        "inline_code": (
            r"`(.*?)`",
            r"<INLINE_CODE>\1<INLINE_CODE>",
        ),  # 인라인 코드 (ex: `코드`) - 코드 텍스트 유지하고 <inline_code>
    }

    # 각 마크다운 문법에 해당하는 부분을 태그로 대체하는 과정
    for pattern, replacement in markdown_patterns.values():
        example["context"] = re.sub(
            pattern, replacement, example["context"], flags=re.MULTILINE
        )  # 패턴에 해당하는 부분을 태그로 대체

    return example
