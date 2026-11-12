import json

from wsp.io import read_jsonl


def test_jsonl_does_not_split_unicode_separators_inside_valid_json_strings(tmp_path):
    rows = [{"problem": "first\u2028second\u0085third"}, {"problem": "another"}]
    path = tmp_path / "practice.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    assert read_jsonl(path) == rows
