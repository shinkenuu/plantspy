from copy import deepcopy
from dsp import passages2text
from dspy.primitives.example import Example
from dspy.primitives.prediction import Prediction
from dspy.primitives.program import Module
from dspy.predict import Predict
from dspy.retrieve import Retrieve
from dspy.signatures import Signature
from dspy.signatures.field import OutputField, InputField
from dspy.signatures.signature import ensure_signature

_FINAL_ACTION_NAME = "Finish"


def _parse_thought(*args, **kwargs):
    print("~~~~~~~~~~~~~~~PARSE THOUGHT~~~~~~~~~~~~~~~")


def _parse_act(*args, **kwargs):
    print("~~~~~~~~~~~~~~~PARSE ACT~~~~~~~~~~~~~~~")


class ReAct(Module):
    def __init__(
        self,
        signature,
        *,
        max_iters: int = 5,
        retrievers: list[Retrieve] = None,
    ):
        super().__init__()
        self.signature = ensure_signature(signature)
        self.max_iters = max_iters

        self.input_fields = self.signature.input_fields
        self.output_fields = self.signature.output_fields

        assert len(self.output_fields) == 1, "ReAct only supports one output field."

        joined_inputs = ", ".join([f"`{k}`" for k in self.input_fields.keys()])
        joined_outputs = ", ".join([f"`{k}`" for k in self.output_fields.keys()])

        retrievers = retrievers or [Retrieve(k=3)]  # maybe get from dspy.config?
        self.tools = self._generate_tools(retrievers=retrievers, outputs=joined_outputs)

        self.instructions = self._generate_instructions(
            tools=self.tools, joined_inputs=joined_inputs, joined_outputs=joined_outputs
        )

        self.reactors = self._generate_reactors(
            n=max_iters,
            input_fields=self.input_fields,
            tools=self.tools,
            instructions=self.instructions,
        )
        # for react_iteration in range(1, max_iters + 1):
        #     signature = Signature(
        #         self._generate_signature(
        #             tools=self.tools, input_fields=self.inputs_fields, iters=react_iteration
        #         ),
        #         self.instructions,
        #     )
        #     prediction = Predict(signature)
        #     self.thinkers.append(prediction)

    @staticmethod
    def _generate_tools(retrievers: list[Retrieve], outputs: str):
        finish_tool = Example(
            name=_FINAL_ACTION_NAME,
            input_variable=outputs.strip("`"),
            desc=f"returns the final {outputs} and finishes the task",
        )

        tools = retrievers + [finish_tool]
        tools_by_name = {tool.name: tool for tool in tools}
        return tools_by_name

    @staticmethod
    def _generate_instructions(
        tools: dict[str, Retrieve | Example], joined_inputs, joined_outputs
    ):
        instructions = [
            f"You will be given {joined_inputs} and you will respond with {joined_outputs}.",
            "To do this, you will interleave Thought, Action, and Observation steps.",
            "Thought can reason about the current situation, and Action can be the following types:\n",
        ]

        for idx, tool in enumerate(tools):
            tool = tools[tool]
            instructions.append(
                f"({idx+1}) {tool.name}[{tool.input_variable}], which {tool.desc}",
            )

        instructions = "\n".join(instructions)
        return instructions

    @classmethod
    def _generate_reactors(
        cls,
        n: int,
        input_fields: dict[str, InputField],
        tools: dict[str, Retrieve | Example],
        instructions: str,
    ):
        reactors = []
        for n_reaction in range(1, n + 1):
            react_signature = cls._generate_react_signature(
                tools=tools,
                input_fields=input_fields,
                iters=n_reaction,
                instructions=instructions,
            )
            reactor = Predict(react_signature)
            reactors.append(reactor)
        # thinkers = [
        #     Predict(Signature(cls._generate_signature(i), instructions))
        #     for i in range(1, max_iters + 1)
        # ]
        return reactors

    @staticmethod
    def _generate_react_signature(
        tools: dict[str, Retrieve | Example],
        input_fields: dict[str, InputField],
        iters: int,
        instructions: str,
    ) -> Signature:
        steps: dict[str, InputField | OutputField] = deepcopy(input_fields)

        iterative_actions = {
            tool.name: f"{tool.name}[{tool.input_variable}]" for tool in tools.values()
        }
        action_finish = iterative_actions.pop(_FINAL_ACTION_NAME)
        joined_iterative_actions = ", ".join(iterative_actions.values())
        action_desc = (
            f"always either {joined_iterative_actions} or, when done, {action_finish}"
        )

        for iteration in range(1, iters + 1):
            steps[f"Thought_{iteration}"] = OutputField(
                prefix=f"Thought {iteration}:",
                desc="next steps to take based on lastest observation",
                # format=_parse_thought,
                parser=_parse_thought,
            )

            steps[f"Action_{iteration}"] = OutputField(
                prefix=f"Action {iteration}:",
                desc=action_desc,
                # format=_parse_act,
                parser=_parse_act,
            )

            if iteration < iters:
                steps[f"Observation_{iteration}"] = OutputField(
                    prefix=f"Observation {iteration}:",
                    desc="observations from latest action",
                    format=passages2text,
                )

        reactor_signature = Signature(steps, instructions)
        return reactor_signature

    @staticmethod
    def _parse_action(action: str):
        action_name, action_arg = action.strip().split("\n")[0].split("[", 1)
        action_arg = action_arg.split("]", 1)[0]
        return action_name, action_arg

    def act(self, reaction: Prediction, hop: int):
        try:
            action = reaction[f"Action_{hop+1}"]
            action_name, action_arg = self._parse_action(action)
            clean_action = f"{action_name}[{action_arg}]"
            reaction[f"Action_{hop+1}"] = clean_action

            if action_name == "Finish":
                return action_arg

            action_tool = self.tools[action_name]
            reaction[f"Observation_{hop+1}"] = action_tool(action_arg).passages

        except Exception as e:
            reaction[f"Observation_{hop+1}"] = (
                "Failed to parse action. Bad formatting or incorrect action name."
            )
            raise e

    def forward(self, **kwargs):
        reactor_kwargs = {
            key: kwargs[key] for key in self.input_fields.keys() if key in kwargs
        }

        for hop in range(self.max_iters):
            # with dspy.settings.context(show_guidelines=(i <= 2)):
            hop_reactor = self.reactors[hop]
            reaction = hop_reactor(**reactor_kwargs)

            if action_val := self.act(reaction, hop):
                break

            reactor_kwargs.update(reaction)

        # assumes only 1 output field for now - TODO: handling for multiple output fields
        output_field_name = list(self.output_fields.keys())[0]
        return Prediction(**{output_field_name: action_val or ""})
