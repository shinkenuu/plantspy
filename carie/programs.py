import dspy

from react import ReAct
from .signatures import Assist, AssistComparison
from .tools import ExaminePlant, ReadPlantSensor, ListPlants  # , WebSearch


class Carie(dspy.Module):
    def __init__(self, max_iters: int = 8, num_results: int = None):
        super().__init__()

        retrievers = [
            ExaminePlant(),
            ReadPlantSensor(),
            ListPlants(),
            # WebSearch()
        ]
        self.generate_reasoning = ReAct(
            Assist, max_iters=max_iters, retrievers=retrievers
        )

    def forward(self, task):
        reasoning = self.generate_reasoning(task=task)
        return reasoning


class AssistanceComparator(dspy.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compare_assistances = dspy.ChainOfThought(AssistComparison)

    def forward(
        self, query: str, gold_assistance: str, actual_assistance: str
    ) -> dspy.Prediction:
        prediction = self.compare_assistances(
            query=query,
            gold_assistance=gold_assistance,
            actual_assistance=actual_assistance,
        )

        return prediction
