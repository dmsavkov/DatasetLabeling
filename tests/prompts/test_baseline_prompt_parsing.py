from src.prompts.baseline import extract_json_array, parse_batch_predictions, strip_markdown_fences


def test_strip_markdown_fences() -> None:
    s = "```json\n[{\"id\":\"1\",\"label\":\"a\"}]\n```"
    assert strip_markdown_fences(s) == '[{"id":"1","label":"a"}]'


def test_extract_json_array_from_wrapped_text() -> None:
    s = "some text\n```json\n[{\"id\":\"1\",\"label\":\"a\"}]\n```\nmore"
    assert extract_json_array(s) == '[{"id":"1","label":"a"}]'


def test_parse_batch_predictions_normalizes_labels() -> None:
    out = parse_batch_predictions(
        '[{"id":"x","label":"A"},{"id":"y","label":"b"},{"id":"z","label":"nope"}]',
        allowed_labels=["a", "b"],
    )
    assert out["x"] == "a"
    assert out["y"] == "b"
    assert out["z"] is None


def test_parse_batch_predictions_accepts_dict_wrapper() -> None:
    out = parse_batch_predictions(
        '{"predictions":[{"id":"x","label":"A"},{"id":"y","label":"b"}]}',
        allowed_labels=["a", "b"],
    )
    assert out["x"] == "a"
    assert out["y"] == "b"

