import pytest

from llikert.server.errors import ServiceError
from llikert.server.prompt import build_messages, prompt_warnings
from llikert.server.task import parse_task

from .conftest import numeric_task_dict, task_dict

DEFAULT_SYSTEM = "Classify the text's primary communicative function.\n\nResponse codes:\nA = description\nB = question\nC = request"
ANSWER = "Answer with exactly one of the response codes listed above and nothing else."


def messages(task_obj, item="Where is it?"):
    return build_messages(parse_task(task_obj).canonical(), item)


def test_default_format_puts_the_answer_instruction_after_the_item():
    assert messages(task_dict()) == [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": "Text:\n<text>\nWhere is it?\n</text>\n\n" + ANSWER},
    ]


def test_answer_instruction_comes_from_the_task():
    msgs = messages(task_dict(answer_instruction="Reply with one letter."))
    assert msgs[1]["content"].endswith("\n\nReply with one letter.")
    assert messages(task_dict(answer_instruction=""))[1]["content"] == "Text:\n<text>\nWhere is it?\n</text>\n\n"


def test_defaults_are_filled_in_canonical_task():
    canonical = parse_task(task_dict()).canonical()
    assert canonical["answer_instruction"] == ANSWER
    assert canonical["prompt"] == {
        "system": "{instructions}\n\n{scale}",
        "user": "Text:\n<text>\n{item}\n</text>\n\n{answer_instruction}",
        "scale": "Response codes:\n{codes}",
        "code": "{response} = {label}",
        "code_separator": "\n",
    }


def test_instructions_after_the_item_without_system_message():
    task = task_dict(prompt={"system": None, "user": "{item}\n\n{instructions}\n{scale}\n{answer_instruction}"})
    assert messages(task) == [
        {
            "role": "user",
            "content": "Where is it?\n\nClassify the text's primary communicative function.\nResponse codes:\nA = description\n"
            "B = question\nC = request\nAnswer with exactly one of the response codes listed above and nothing else.",
        }
    ]


def test_questionnaire_style_minimal_prompt():
    task = numeric_task_dict(
        name="Big Five: extraversion",
        instructions="You are completing a personality questionnaire. Rate how well each statement describes you.",
        answer_instruction="Reply with the number only.",
        prompt={
            "system": "{instructions}\n{scale}",
            "user": "{item}\n\n{answer_instruction}",
            "scale": "{codes}",
            "code": "{response} = {label}",
            "code_separator": "; ",
        },
    )
    msgs = messages(task, "I am the life of the party.")
    assert msgs[1] == {"role": "user", "content": "I am the life of the party.\n\nReply with the number only."}
    assert msgs[0]["content"].endswith("1 = very negative; 2 = negative; 3 = neutral; 4 = positive; 5 = very positive")


def test_value_and_id_placeholders_and_number_formatting():
    task = numeric_task_dict(prompt={"code": "{response}: {label} (value {value}, id {id})", "code_separator": " | "})
    for c, v in zip(task["categories"], [-2.5, 0, 0.1, 1e21, 3]):
        c["value"] = v
    content = messages(task)[0]["content"]
    assert "1: very negative (value -2.5, id s1) | 2: negative (value 0, id s2) | 3: neutral (value 0.1, id s3)" in content
    assert "(value 1e+21, id s4)" in content and "(value 3, id s5)" in content


def test_examples_use_the_user_template_and_response_codes():
    task = task_dict(examples=[{"item": "Close it.", "category_id": "request"}], prompt={"user": "Item: {item}"})
    assert [(m["role"], m["content"]) for m in messages(task)[1:]] == [
        ("user", "Item: Close it."), ("assistant", "C"), ("user", "Item: Where is it?"),
    ]


def test_substitution_is_single_pass_and_braces_escape():
    task = task_dict(instructions="Use {scale} literally", prompt={"user": "{{literal}} {item}"})
    msgs = messages(task, "an item with {scale} and {item} inside")
    assert msgs[0]["content"].startswith("Use {scale} literally\n\nResponse codes:")
    assert msgs[1]["content"] == "{literal} an item with {scale} and {item} inside"


@pytest.mark.parametrize(
    "prompt,fragment",
    [
        ({"user": "no item here"}, "{item} exactly once"),
        ({"user": "{item} {item}"}, "{item} exactly once"),
        ({"system": "{item}"}, "must not contain {item}"),
        ({"user": "{item} {unknown}"}, "unknown placeholder {unknown}"),
        ({"user": "{item} {"}, "unmatched brace"),
        ({"scale": "no codes"}, "{codes} exactly once"),
        ({"code": "{label}"}, "must contain {response}"),
        ({"code": "{response} {value}"}, "no values"),
        ({"scale": "{codes} {item}"}, "unknown placeholder {item}"),
        ({"extra": "x"}, "Extra inputs"),
        ({"answer_instruction": "Answer with one code."}, "Extra inputs"),
    ],
)
def test_invalid_prompt_formats(prompt, fragment):
    with pytest.raises(ServiceError) as info:
        parse_task(task_dict(prompt=prompt))
    assert info.value.code == "invalid_task"
    assert any(fragment in (e["message"] or "") for e in info.value.details["errors"]), info.value.details


def test_empty_instructions_are_allowed_with_a_warning():
    task = parse_task(task_dict(instructions=""))
    assert [w["code"] for w in prompt_warnings(task.canonical())] == ["empty_instructions"]


def test_warnings_for_unused_parts():
    task = parse_task(task_dict(prompt={"system": None, "user": "{item}"}))
    codes = [w["code"] for w in prompt_warnings(task.canonical())]
    assert codes == ["unused_instructions", "unused_answer_instruction", "scale_not_in_prompt"]
    assert prompt_warnings(parse_task(task_dict()).canonical()) == []


def test_prompt_format_changes_task_identity():
    assert parse_task(task_dict()).identity() != parse_task(task_dict(prompt={"code": "{response}: {label}"})).identity()
    assert parse_task(task_dict()).identity() != parse_task(task_dict(answer_instruction="Reply with one letter.")).identity()
    assert parse_task(task_dict()).identity() == parse_task(task_dict(prompt={})).identity()


def test_prepare_returns_messages_and_warnings(service):
    out = service.prepare(task_dict(prompt={"system": None, "user": "{item}"}), preview_text="Where?")
    assert out["preview"]["text"]["messages"] == [{"role": "user", "content": "Where?"}]
    assert "scale_not_in_prompt" in [w["code"] for w in out["diagnostics"]["warnings"]]
