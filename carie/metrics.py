from numpy import mean, median

from carie.programs import is_semantically_similar


def evaluate_carie(gold, prediction, *args, **kwargs):
    if not prediction.result.strip():
        return 0

    assert gold.task == prediction.task

    thought_score = _score_thoughts(gold=gold, prediction=prediction)
    action_score = _score_actions(gold=gold, prediction=prediction)
    result_score = _score_result(gold=gold, prediction=prediction)
    scores = [thought_score, action_score, result_score]
    score = median(scores)

    return score


def _score_thoughts(gold, prediction) -> float:
    prediction_thought_hops = sum(
        [1 for step_name in prediction if "thought" in step_name.lower()]
    )

    semantic_similarity_scores = []

    for hop in range(1, prediction_thought_hops + 1):
        hop_step_name = f"Thought_{hop}"

        try:
            gold_thought = gold[hop_step_name]
        except KeyError:
            semantic_similarity_scores.append(0)
            continue

        predicted_thought = prediction[hop_step_name]
        semantically_similar = is_semantically_similar(gold_thought, predicted_thought)
        semantic_similarity_score = 1.0 if semantically_similar else 0.0
        semantic_similarity_scores.append(semantic_similarity_score)

    score = mean(semantic_similarity_scores)
    return score


def _score_actions(gold, prediction) -> float:
    actions = [step_name for step_name in prediction if "action" in step_name.lower()]

    has_duplicates = len(actions) > len(set(actions))
    score = 0.0 if has_duplicates else 1.0

    return score


def _score_result(gold, prediction) -> float:
    gold_result = gold.result
    prediction_result = prediction.result

    semantically_similar = is_semantically_similar(gold_result, prediction_result)
    score = 1.0 if semantically_similar else 0.0

    return score
