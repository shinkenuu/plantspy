from dspy.teleprompt import BootstrapFewShot

from .metrics import is_correct


def optimize_boostrap_fewshot(
    program, trainset, valset, max_bootstrapped_demos: int = 5, **kwargs
):
    teleprompt = BootstrapFewShot(
        metric=is_correct,
        max_bootstrapped_demos=max_bootstrapped_demos,
        **kwargs,
    )

    bootstrap = teleprompt.compile(program, trainset=trainset, valset=valset)
    return bootstrap
