from csv import DictReader
from random import shuffle

from dspy import Example


_FIELD_MAP = {
    "task": "task",
    "result": "result",
    "thought_1": "Thought_1",
    "action_1": "Action_1",
    "observation_1": "Observation_1",
    "thought_2": "Thought_2",
    "action_2": "Action_2",
    "observation_2": "Observation_2",
    "thought_3": "Thought_3",
    "action_3": "Action_3",
    "observation_3": "Observation_3",
}


def _read_csv(
    file_path: str,
    field_map=_FIELD_MAP,
):
    with open(file_path) as file:
        reader = DictReader(file)
        for row in reader:
            example = {
                output_field: row.get(input_field)
                for input_field, output_field in field_map.items()
            }
            yield example


def _load_sensor_csv(file_names: list[str] = None):
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


def split(rows: list, test_size: float = 0.3):
    n_test = int(len(rows) * test_size)
    shuffle(rows)
    return rows[:n_test], rows[n_test:]


def load_examples(n_examples: int = None, test_size: float = 0.3):
    rows = list(_load_sensor_csv())
    # rows = list(_load_prototype_examples())
    n_examples = n_examples or len(rows)

    if len(rows) < n_examples:
        raise ValueError(
            f"Not enough examples. Desired {n_examples}, actual {len(rows)}"
        )

    trainset, testset = split(rows, test_size=test_size)
    valset, testset = split(testset, test_size=0.5)

    trainset = [Example(row).with_inputs("task") for row in trainset]
    valset = [Example(row).with_inputs("task") for row in valset]
    testset = [Example(row).with_inputs("task") for row in testset]

    return trainset, valset, testset
