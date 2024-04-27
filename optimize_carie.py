from datetime import datetime

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot

from carie.data import load_examples
from carie.lm import get_llama3
from carie.programs import Carie
from carie.metrics import evaluate_carie


lm = get_llama3()
dspy.settings.configure(lm=lm)


# Load data
trainset, valset, testset = load_examples(test_size=0.33)

# Optimize
bs_few_shot = BootstrapFewShot(
    metric=evaluate_carie,
    metric_threshold=0.5,
    max_bootstrapped_demos=2,
    max_labeled_demos=2,
    max_errors=2,
)

carie = Carie()
bs_few_shot_carie = bs_few_shot.compile(carie, trainset=trainset, valset=valset)


# Evaluate
evaluator = Evaluate(
    devset=testset,
    metric=evaluate_carie,
    num_threads=16,
    display_progress=True,
    display_table=0,
)

base_score = evaluator(carie)
print("Base score: ", base_score)

bs_few_shot_score = evaluator(bs_few_shot_carie)
print("Few-shot score: ", bs_few_shot_score)

# Save
filename = f"compiled/carie_bs_few_shot_{bs_few_shot_score}_llama3_{datetime.now().isoformat()}.json"
bs_few_shot_carie.save(filename)
