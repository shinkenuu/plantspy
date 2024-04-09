from copy import deepcopy
from functools import partial
from typing import Literal

from dspy.primitives.example import Example
from dspy.primitives.prediction import Prediction
from dspy.primitives.program import Module
from dspy.predict import Predict
from dspy.retrieve import Retrieve
from dspy.signatures import Signature
from dspy.signatures.field import OutputField, InputField
from dspy.signatures.signature import ensure_signature

_FINISH_ACTION_NAME = "Finish"


def _generate_tools(retrievers: list[Retrieve], outputs: str):
    finish_tool = Example(
        name=_FINISH_ACTION_NAME,
        input_variable=outputs.strip("`"),
        desc=f"returns the final {outputs} and finishes the task",
    )

    tools = retrievers + [finish_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    return tools_by_name


def _generate_instructions(
    tools: dict[str, Retrieve | Example], joined_inputs, joined_outputs
):
    instructions = [
        f"You will be given {joined_inputs} and you will respond with {joined_outputs}.",
        "To do this, you will interleave Thought, Action, and Observation steps.",
        "Thought can reason about the current situation, Observation contains previous Action outputs and Action can be the following types:\n",
    ]

    for idx, tool in enumerate(tools):
        tool = tools[tool]
        instructions.append(
            f"({idx+1}) {tool.name}[{tool.input_variable}], which {tool.desc}",
        )

    joined_instructions = "\n".join(instructions)
    return joined_instructions


def _generate_planners(
    n: int,
    input_fields: dict[str, InputField],
    tools: dict[str, Retrieve | Example],
    instructions: str,
):
    planners = []
    for n_reaction in range(1, n + 1):
        react_signature = _generate_react_signature(
            tools=tools,
            input_fields=input_fields,
            hops=n_reaction,
            instructions=instructions,
        )
        reactor = Predict(react_signature)
        planners.append(reactor)

    return planners


def _generate_react_signature(
    tools: dict[str, Retrieve | Example],
    input_fields: dict[str, InputField],
    hops: int,
    instructions: str,
) -> Signature:
    steps: dict[str, InputField | OutputField] = deepcopy(input_fields)

    action_templates = {
        tool.name: f"{tool.name}[{tool.input_variable}]" for tool in tools.values()
    }
    action_finish_template = action_templates.pop(_FINISH_ACTION_NAME)
    joined_actions_templates = ", ".join(action_templates.values())

    for hop in range(1, hops + 1):
        steps[f"Thought_{hop}"] = OutputField(
            prefix=f"Thought {hop}:",
            desc="next best step towards finishing the task",
            format=partial(_clean_thought, hop=hop),
        )

        steps[f"Action_{hop}"] = OutputField(
            prefix=f"Action {hop}:",
            desc=f"always either {joined_actions_templates} or, when done, {action_finish_template}",
            format=_clean_action,
        )

        if hop < hops:
            steps[f"Observation_{hop}"] = OutputField(
                prefix=f"Observation {hop}:",
                desc=f"results from Action {hop}",
                format=_clean_observation,
            )

    reactor_signature = Signature(steps, instructions)
    return reactor_signature


def _clean_thought(thought: str, hop: int) -> str:
    stop = f"Action {hop +1}:"
    stop_index = thought.find(stop)

    if stop_index > 0:
        clean_thought = thought[:stop_index].strip()
    else:
        clean_thought = thought.strip()

    return clean_thought


def _parse_action(action: str) -> str:
    action_name, action_arg = action.strip().split("\n")[0].split("[", 1)
    action_arg = action_arg.split("]", 1)[0]
    return action_name, action_arg


def _format_action(action_name: str, action_arg: str) -> str:
    formatted_action = f"{action_name}[{action_arg}]"
    return formatted_action


def _clean_action(action: str) -> str:
    action_name, action_arg = _parse_action(action)
    clean_action = _format_action(action_name, action_arg)
    return clean_action


def _clean_observation(passages: list[str] | str) -> str:
    if isinstance(passages, str):
        return passages

    clean_observations = [passage.strip() for passage in passages]
    clean_observation = "\n".join(clean_observations)
    return clean_observation


def _get_latest_step(
    reaction: Prediction, step_name: Literal["thought", "action", "observation"]
) -> tuple[int, str]:
    lower_step_name = step_name.lower()
    steps = {}

    for reaction_step in reaction:
        if lower_step_name not in reaction_step.lower():
            continue

        # Thought_1 | Action_1 | Observation_1
        hop = int(reaction_step.split("_")[-1])
        steps[hop] = reaction_step

    last_hop = max(steps.keys())
    return last_hop, steps[last_hop]


def get_reactions_by_step(
    reactions, step_name: Literal["thought", "action", "observation"]
):
    reactions_of_step = {
        reaction_step_name: reactions[reaction_step_name]
        for reaction_step_name in reactions
        if step_name in reaction_step_name.lower()
    }

    return reactions_of_step


def is_duplicate_action(action: str, reactions: dict[str, str]) -> bool:
    action_reactions = get_reactions_by_step(reactions, step_name="action")
    lower_action_values = {
        action_value.lower() for action_value in action_reactions.values()
    }
    lower_action_value = action.lower()

    is_duplicate_action = lower_action_value in lower_action_values
    return is_duplicate_action


class ReAct(Module):
    def __init__(
        self,
        signature,
        *,
        max_hops: int = 5,
        retrievers: list[Retrieve] = None,
    ):
        super().__init__()
        self.signature = ensure_signature(signature)
        self.max_hops = max_hops

        self.input_fields = self.signature.input_fields
        self.output_fields = self.signature.output_fields

        assert len(self.output_fields) == 1, "ReAct only supports one output field."

        joined_inputs = ", ".join([f"`{k}`" for k in self.input_fields.keys()])
        joined_outputs = ", ".join([f"`{k}`" for k in self.output_fields.keys()])

        retrievers = retrievers or [Retrieve(k=3)]
        self.tools = _generate_tools(retrievers=retrievers, outputs=joined_outputs)

        self.instructions = _generate_instructions(
            tools=self.tools, joined_inputs=joined_inputs, joined_outputs=joined_outputs
        )

        self.planners = _generate_planners(
            n=max_hops,
            input_fields=self.input_fields,
            tools=self.tools,
            instructions=self.instructions,
        )

    def act_and_observe(self, reaction: Prediction):
        step_n, latest_action_step = _get_latest_step(reaction, step_name="action")
        latest_action = reaction[latest_action_step]

        try:
            action_name, action_arg = _parse_action(latest_action)
            reaction[latest_action_step] = _format_action(action_name, action_arg)

            if action_name == "Finish":
                return action_arg

            action_tool = self.tools[action_name]
            reaction[f"Observation_{step_n}"] = action_tool(action_arg).passages

        except Exception:
            reaction[f"Observation_{step_n}"] = (
                "Failed to parse action. Bad formatting or incorrect action name."
            )

    def forward(self, **kwargs):
        reactions = {
            key: kwargs[key] for key in self.input_fields.keys() if key in kwargs
        }

        for idx, planner in enumerate(self.planners):
            reaction = planner(**reactions)

            # Suggest(
            #     is_duplicate_action(reaction[f"Action_{idx+1}"], reactions),
            #     "This action has been used before with its results in observations. Re-examine observations and try again.",
            # )

            final = self.act_and_observe(reaction)
            reactions.update(reaction)

            if final:
                break

        # assumes only 1 output field for now - TODO: handling for multiple output fields
        output_field_name = list(self.output_fields.keys())[0]
        prediction_kwargs = {output_field_name: final or "", **reactions}
        prediction = Prediction(**prediction_kwargs)
        return prediction
