from goro_llm.experiment import load_benchmark


def test_benchmark_csv_keeps_formula_columns_intact():
    rows = load_benchmark("data/benchmark.csv")
    assert len(rows) == 4
    inverse = next(row for row in rows if row["id"] == "matrix_inverse_2x2")
    assert inverse["fact"].endswith("[[d,−b],[−c,a]]")
    assert inverse["protected"] == "ひく"
