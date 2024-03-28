import dspy

dspy.ReAct


class Assist(dspy.Signature):
    """Answer questions and perform actions regarding in-house anthropomorphed plants"""

    task = dspy.InputField(desc="a need to be fulfilled")
    result = dspy.OutputField(
        desc="either a success or failure result of task",
    )


class AssistComparison(dspy.Signature):
    """Compare 2 assistance results and score the actual assistance quality against the gold assistance"""

    query = dspy.InputField()
    actual_assistance = dspy.InputField()
    gold_assistance = dspy.InputField()

    correctness_score = dspy.OutputField(
        desc="Between 0 and 1, the higher the better",
    )
    tonality_score = dspy.OutputField(
        desc="Between 0 and 1, the higher the better",
    )
