import dspy

from carie.data import split_examples
from carie.lm import get_lm
from carie.programs import Carie
from carie.optimizers import optimize_boostrap_fewshot

dspy.settings.configure(lm=get_lm())
carie = Carie()

trainset, valset = split_examples(test_proportion=0.25)

optimal_program = optimize_boostrap_fewshot(
    program=carie, trainset=trainset, valset=valset
)
optimal_program.save("bs_fs_carie.json")
