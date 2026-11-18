import pandas as pd

from src.atlas.flip import cache_files, flip_cache
from src.atlas.readers import Matched, Unmatched


def write(path, rows):
    pd.DataFrame({"year": ["2020", "2021", "2022"], "v": ["1", "2", "3"]}).head(rows).to_parquet(path, index=False)


def test_each_dataset_uses_its_best_cache_file_and_the_rest_are_accounted_for(tmp_path):
    write(tmp_path / "d_a_10000.parquet", 3)
    write(tmp_path / "d_a_all.parquet", 3)
    write(tmp_path / "d_b_5000.parquet", 3)
    write(tmp_path / "d_b_10000.parquet", 3)
    files = cache_files(tmp_path)
    assert list(files) == ["d_a", "d_b"]
    assert files["d_a"][0][1].name == "d_a_all.parquet"
    assert files["d_b"][0][1].name == "d_b_10000.parquet"
    assert sum(len(v) for v in files.values()) == 4


def test_a_file_at_its_row_cap_is_flagged_truncated_and_a_missing_dataset_is_unmatched(tmp_path):
    write(tmp_path / "d_a_3.parquet", 3)
    write(tmp_path / "d_b_all.parquet", 3)
    out = dict((r.dataset_id, (r, t)) for r, t in flip_cache(["d_a", "d_b", "d_missing"], cache_files(tmp_path), {}))
    assert isinstance(out["d_a"][0], Matched) and out["d_a"][1] is True
    assert isinstance(out["d_b"][0], Matched) and out["d_b"][1] is False
    assert isinstance(out["d_missing"][0], Unmatched) and out["d_missing"][0].reason == "not_in_cache"
