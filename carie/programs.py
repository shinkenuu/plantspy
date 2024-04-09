import dspy

from react import ReAct
from .tools import ExaminePlant, ReadPlantSensor, ListPlants  # , WebSearch
from text_parsers import parse_boolean


class CarieSignature(dspy.Signature):
    """Perform tasks using actions in order to care for household plants"""

    task = dspy.InputField(desc="the question or need we will perform for")
    result = dspy.OutputField(
        desc="either the success or the failure result of task",
    )


class Carie(dspy.Module):
    def __init__(self, max_hops: int = 8):
        super().__init__()

        self.retrievers = [
            ExaminePlant(),
            ReadPlantSensor(),
            ListPlants(),
            # WebSearch()
        ]
        self.generate_reasoning = ReAct(
            CarieSignature, max_hops=max_hops, retrievers=self.retrievers
        )

    def forward(self, task):
        reasoning = self.generate_reasoning(task=task)
        return reasoning


class SemanticSimilarity(dspy.Signature):
    """Verify that two texts are semantically similar"""

    text_1 = dspy.InputField()
    text_2 = dspy.InputField()
    is_semantically_similar = dspy.OutputField(
        desc="Either true or false", format=lambda x: x.split("\n")[0]
    )


def is_semantically_similar(text_1: str, text_2: str):
    program = dspy.ChainOfThought(SemanticSimilarity)
    prediction = program(text_1=text_1, text_2=text_2)
    return parse_boolean(prediction.is_semantically_similar)
