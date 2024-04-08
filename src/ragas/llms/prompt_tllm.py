# @title
import json
import sys
sys.path.insert(0,r'./')
import pprint
from typing import List, Dict
from dataclasses import dataclass, asdict, fields


@dataclass
class AdvanceInstructSample:
    """
    A single training/test example for the Instruct dataset.
    """
    qas_id: str
    system_prompt: str

    question_text: str

    orig_answer_texts: str = None
    answer_lengths: int = None

    def __post_init__(self) -> None:
        # Post validate
        self.answer_lengths = len(self.orig_answer_texts) if self.orig_answer_texts is not None else None

    def __str__(self) -> str:
        return self.__repr__

    @property
    def __repr__(self) -> str:
        s = ""
        s += f"\n Question id: {self.qas_id}"
        s += f"\n System prompt: {self.system_prompt}"
        s += f"\n Question: {self.question_text}"
        if self.orig_answer_texts:
            s += f"\n Answer text: {self.orig_answer_texts}"
            s += f"\n Answer length: {self.answer_lengths}"

        return s

    @property
    def get_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def get_keys() -> List[str]:
        all_fields = fields(AdvanceInstructSample)
        return [v.name for v in all_fields]

    @property
    def get_dict_str(self, indent: int=4) -> None:
        pp = pprint.PrettyPrinter(indent=indent)
        pp.pprint(self.get_dict)

    def get_example(self,
                    inputs_column: str="prompt",
                    targets_column: str="target",
                    system_prefix: str="",
                    question_prefix: str="####### Instruction:",
                    response_prefix: str="%%%%%%% Response:",
                    is_training: bool=True,
                    do_perplexity_eval: bool=False,
                    do_generative_eval: bool=False,
                    task_type: str=None,
                    ) -> Dict:
        assert task_type, "Please specified the task type inorder to get the example"

        system_msg = ' ' + system_prefix + '\n' + self.system_prompt + "\n\n"
        question_msg = question_prefix + '\n' + self.question_text + "\n\n"
        prompt = system_msg + ' ' + question_msg
        label = self.orig_answer_texts + "\n"

        if task_type == "SEQ_2_SEQ_LM":
            return {inputs_column: prompt,
                    targets_column: label}
        elif task_type == "CAUSAL_LM":
            if is_training:
                return {inputs_column: prompt + ' ' + response_prefix + '\n' + label}

            example_dict = {}
            # The perplexity field is for perplexity evaluation, which needed the full prompt and label
            # while the inputs_column only have prompt and response_prefix for model.generate evaluation
            if do_generative_eval:
                example_dict[inputs_column] = prompt + ' ' + response_prefix + '\n'
                example_dict[targets_column] = label

            if do_perplexity_eval:
                example_dict["perplexity"] = prompt + ' ' + response_prefix + '\n' + label

            if not bool(example_dict):
                raise "Evaluation files is provided but don't know what to do with them..."

            return example_dict
        else:
            raise f"This task type {task_type} is not support"


if __name__ == "__main__":
    example8 = AdvanceInstructSample(qas_id="8", question_text="What do cats eat?",
                                    orig_answer_texts="meat and fish", system_prompt="Hi")

    print(example8.get_example(is_training=True, task_type="CAUSAL_LM"))
    example6 = AdvanceInstructSample(qas_id="6", question_text="What is the meaning of existence?",
                                     orig_answer_texts="Dying", system_prompt="Hello")
    # print(example6)
    print(example6.get_example(is_training=True, task_type="SEQ_2_SEQ_LM"))

    # @title
    json_gbnf = r"""root   ::= customObject
    value  ::= object | array | string | number | ("true" | "false" | "null") ws

    customObject ::=
    "{" ws (
        "\"reasoning\":" ws value ","
        ws "\"intent\":" ws value
    )? "}" ws

    object ::=
    "{" ws (
                string ":" ws value
        ("," ws string ":" ws value)*
    )? "}" ws

    array  ::=
    "[" ws (
                value
        ("," ws value)*
    )? "]" ws

    string ::=
    "\"" (
        [^"\\\x7F\x00-\x1F] |
        "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) # escapes
    )* "\"" ws

    number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

    # Optional space: by convention, applied in this grammar after literal chars when allowed
    ws ::= ([ \t\n] ws)?"""

    # Load the json_gbnf object
    # (Assume it is already defined and contains the data you want to save)

    # Open a file for writing in binary mode
    with open('json.gbnf', 'w') as outfile:
        # Encode the json_gbnf string to bytes and write to the file
        outfile.write(json_gbnf)

    with open('json.gbnf', 'r') as infile:
        json_grammar = infile.read()
    print(json_grammar)