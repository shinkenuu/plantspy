from dspy.evaluate.auto_evaluation import AnswerCorrectness
from dspy.predict import ChainOfThought

_tonality_scorer = ChainOfThought(
    "expected tonality example, evaluating example -> tonality similarity score"
)
_answer_correctness = AnswerCorrectness()


def is_similar_assistance(example, prediction, *args, **kwargs):
    if not prediction.result.strip():
        return 0

    correctness_score = _parse_score(
        _answer_correctness(
            question=example.query,
            gold_answer=example.answer,
            predicted_answer=prediction.result,
        )
    )
    tonality_score = _parse_score(_tonality_scorer(example.query, prediction.result))

    scores = (correctness_score, tonality_score)

    similarity_score = sum(scores) // len(scores)
    return similarity_score


def _parse_score(score_text: str):
    try:
        score = float(score_text.strip().split()[0])
    except Exception:
        return 0

    return score


def is_correct(example, prediction, trace=None, frac=0.7):
    if not prediction.result.strip():
        return 0

    answer_correctness = _answer_correctness(
        question=example.query,
        gold_answer=example.answer,
        predicted_answer=prediction.result,
    )
    is_correct = answer_correctness.is_correct.strip().lower().startswith("true")
    # tonality = .96 # tone matches persona
    # relevancy = .99 # relevancy to fulfill query
    # return sum([factualness, correctness, tonality, relevancy]) > 3.6
    return is_correct
