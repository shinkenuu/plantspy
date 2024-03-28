from csv import DictReader
from random import shuffle

from dspy import Example


def _read_csv(file_path: str, fieldnames=["query", "answer"]):
    with open(file_path) as file:
        reader = DictReader(file, fieldnames)
        next(reader)  # skip headers
        for row in reader:
            yield {"task": row["query"], "result": row["answer"]}


def _load_csv_examples(file_names: list[str] = None):
    file_dir_path = "./storage/"
    file_names = file_names or [
        "air_humidity",
        "air_temperature",
        "light_level",
        "soil_humidity",
        "soil_ph",
    ]

    for file_name in file_names:
        file_path = file_dir_path + file_name + ".csv"
        yield from _read_csv(file_path=file_path)


def split_examples(n_examples: int = -1, test_proportion: float = 0.3):
    rows = list(_load_csv_examples())
    n_examples = n_examples if n_examples > 0 else len(rows)

    if len(rows) < n_examples:
        raise ValueError(
            f"Not enough examples. Desired {n_examples}, actual {len(rows)}"
        )

    total_test_examples = int(n_examples * test_proportion)
    shuffle(rows)

    trainset = [Example(row).with_inputs("task") for row in rows[total_test_examples:]]
    testset = [Example(row).with_inputs("task") for row in rows[:total_test_examples]]

    return trainset, testset
