from walkforward import load_returns, load_scores


def test_load_scores_round_trips_csv_fixture(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text(
        "fold,path,in_sample,out_of_sample,return\n"
        "0,0,1.0,0.8,0.01\n"
        "0,1,0.9,0.95,0.02\n",
        encoding="utf-8",
    )

    rows = load_scores(path)

    assert rows == [
        {
            "fold": 0.0,
            "path": 0.0,
            "train": 1.0,
            "in_sample": 1.0,
            "test": 0.8,
            "out_of_sample": 0.8,
            "return": 0.01,
        },
        {
            "fold": 0.0,
            "path": 1.0,
            "train": 0.9,
            "in_sample": 0.9,
            "test": 0.95,
            "out_of_sample": 0.95,
            "return": 0.02,
        },
    ]
    assert load_returns(path) == [0.01, 0.02]


def test_load_scores_reads_json_rows(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(
        '[{"fold": 0, "path": "a", "train": 1.0, "test": 0.8}]',
        encoding="utf-8",
    )

    assert load_scores(path)[0]["in_sample"] == 1.0
    assert load_scores(path)[0]["out_of_sample"] == 0.8


def test_load_scores_accepts_utf8_bom_csv(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text(
        "\ufefffold,path,train,test\n0,a,1.0,0.8\n1,b,0.9,0.7\n",
        encoding="utf-8",
    )

    rows = load_scores(path)

    assert [row["fold"] for row in rows] == [0.0, 1.0]
