import os
import re
import json
import dotenv
import argparse
from enum import Enum
from typing import Callable
from collections import defaultdict

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

# from langchain_core.language_models.base import BaseLanguageModel

import utils


class Section(Enum):
    qa = "qa"
    physical_exam = "physical_exam"
    closure = "closure"
    diagnosis = "diagnosis"
    other = "other"


# Function to load the dataset from a given path
def load_data(dataset_path, generation):
    if os.path.isdir(dataset_path):
        # Determine the correct dataset file based on the generation flag
        if generation:
            dataset_path = os.path.join(dataset_path, "generation.json")
        else:
            dataset_path = os.path.join(dataset_path, "evaluation.json")

    print(f"Loading data from {dataset_path}")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Create a lookup dictionary to map sections and case IDs to conversation turns
    lookup = dict(
        zip(
            [section.value for section in Section],
            [defaultdict(dict) for i in range(5)],
        )
    )

    prev_section = Section.qa.value
    prev_case = 1
    max_turn = 0
    for i, element in enumerate(data):
        section = element["section"]
        case_id = int(element["case_id"])
        conversation_turn_id = int(element["conversation_turn_id"])
        lookup[section][case_id][conversation_turn_id] = i

        if prev_section != section or prev_case != case_id:
            lookup[prev_section][prev_case]["max_turn"] = max_turn
            prev_section = section
            prev_case = case_id
            max_turn = 0
        max_turn += 1

    lookup[prev_section][prev_case]["max_turn"] = max_turn

    return data, lookup


# Create a lookup dictionary to map sections and case IDs to conversation turns
def save_result(path: str, result: dict):
    with open(path, "w", encoding="UTF-8") as f:
        json.dump(result, f, indent=2)


# Function to parse a range or a single number from a string
def parse_range(val):
    if val.strip() == "":
        print("Empty range!")
        return None

    # Match a range like "1-44"
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", val)
    if m is not None:
        start = int(m.group(1))
        end = int(m.group(2))
        return start, end

    # Match a single number like "1"
    num = re.fullmatch(r"\s*(\d+)\s*", val)
    if num is not None:
        start = int(num.group(1))
        return start, start

    print("Invalid range!")
    return None


def run_model(
    llm,
    prompt_template: str,
    input_data: dict[str:str],
    prev_processing: Callable = lambda x: x,
    post_processing: Callable = lambda x: x,
    **kwargs,
):
    # if not issubclass(model_class, BaseLanguageModel):
    #     print("only support langchain class")

    prompt = PromptTemplate.from_template(prompt_template)

    # llm = model_class(**kwargs)

    eval_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
    )

    input_data = prev_processing(input_data)

    result = eval_chain.invoke(input_data)["text"]

    return post_processing(
        result,
    )


def llm_as_medical_student(
    dataset_path: str,
    output_dir: str,
    section: str,
    case: str,
    conversation_turn: str = "all",
    llm=None,
    model_parameters=None,
    prompt_template: str = None,
    input_data: dict[str:str] = None,
    pre_processing: Callable = None,
    post_processing: Callable = None,
    **kwargs,
):
    print(f"Running llm as medical student on {section}:")

    post_processing_func = {
        Section.qa.value: lambda x: x,
        Section.closure.value: lambda x: x,
        Section.physical_exam.value: utils.medical_student_physical_exam_post_processing,
        Section.diagnosis.value: utils.medical_student_diagnosis_post_processing,
    }

    dataset, lookup = load_data(dataset_path, generation=True)

    start_case, end_case = (1, 44) if str(case) == "all" else parse_range(str(case))

    for case in range(start_case, end_case + 1):
        if str(conversation_turn) == "all":
            start_conversation_turn = 1
            end_conversation_turn = lookup[section][case]["max_turn"]
        else:
            start_conversation_turn, end_conversation_turn = parse_range(
                str(conversation_turn)
            )
        # if start_conversation_turn == -1 or end_conversation_turn == -1:
        #     start_conversation_turn = 1
        #     max_turn = lookup[section][case]["max_turn"]
        #     end_conversation_turn = 1 if max_turn is None else max_turn

        for turn in range(start_conversation_turn, end_conversation_turn + 1):
            print(f"Running case {case}, turn {turn}")

            index = lookup[section][case][turn]
            data = dataset[index]
            if prompt_template is None:
                prompt_template = data["prompt"]["template"]
            if input_data is None:
                input_data = data["input"]
            if llm is None:
                if model_parameters is None:
                    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
                else:
                    llm = ChatOpenAI(**model_parameters)
            if pre_processing is None:
                pre_processing = lambda x: x
            if post_processing is None:
                post_processing = post_processing_func[section]

            result = run_model(
                llm=llm,
                prompt_template=prompt_template,
                input_data=input_data,
                prev_processing=pre_processing,
                post_processing=post_processing,
                **kwargs,
            )

            # save result
            dataset[index]["output"][llm.model_name] = result

            output_path = os.path.join(os.getcwd(), output_dir)
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
                output_path = os.path.join(output_path, "generation.json")
            save_result(output_path, dataset)

    print("Finish")


def llm_as_examiner(
    dataset_path: str,
    output_dir: str,
    section: str,
    case: str,
    conversation_turn: str = "all",
    llm=None,
    model_parameters=None,
    prompt_template: str = None,
    input_data: dict[str:str] = None,
    student_input_model_name: str = None,
    pre_processing: Callable = None,
    post_processing: Callable = None,
    **kwargs,
):
    print(f"Running llm as examiner on {section}:")

    post_processing_func = {
        Section.qa.value: utils.readable_json,
        Section.physical_exam.value: utils.readable_json,
        Section.closure.value: utils.readable_json,
        Section.diagnosis.value: utils.readable_json,
    }

    dataset, lookup = load_data(dataset_path, generation=False)

    start_case, end_case = (1, 44) if str(case) == "all" else parse_range(str(case))

    for case in range(start_case, end_case + 1):
        if str(conversation_turn) == "all":
            start_conversation_turn = 1
            end_conversation_turn = lookup[section][case]["max_turn"]
        else:
            start_conversation_turn, end_conversation_turn = parse_range(
                str(conversation_turn)
            )

        for turn in range(start_conversation_turn, end_conversation_turn + 1):
            print(f"Running case {case}, turn {turn}")

            index = lookup[section][case][turn]
            data = dataset[index]
            if prompt_template is None:
                prompt_template = data["prompt"]["template"]
            if input_data is None:
                if student_input_model_name is None:
                    print(
                        "Missing model name specified which input data used to evaluate!"
                    )
                    return
                else:
                    # print(json.dumps(data, indent=2))
                    input_data = data["input"][student_input_model_name]
            if llm is None:
                if model_parameters is None:
                    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
                else:
                    llm = ChatOpenAI(**model_parameters)
            if pre_processing is None:
                pre_processing = lambda x: x
            if post_processing is None:
                post_processing = post_processing_func[section]

            result = run_model(
                llm=llm,
                prompt_template=prompt_template,
                input_data=input_data,
                prev_processing=pre_processing,
                post_processing=post_processing,
                **kwargs,
            )

            # save result
            dataset[index]["result"][llm.model_name] = result

            output_path = os.path.join(os.getcwd(), output_dir)
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
                output_path = os.path.join(output_path, "evaluation.json")
            save_result(output_path, dataset)

    print("Finish")


def main(
    task,
    section,
    case,
    turn,
    dataset_path,
    output_dir,
    student_model=None,
    student_input_model=None,
    examiner_model=None,
):
    if task != "student" and task != "examiner" and task != "all":
        print("Invalid task!")
        return

    if task == "student" or task == "all":
        if not student_model:
            print(
                "Missing student model name specifies which model used for generating!"
            )
            return
        llm_as_medical_student(
            dataset_path,
            output_dir,
            section,
            case,
            turn,
            model_parameters={"model_name": student_model, "temperature": 0},
            # prompt_template="",
        )
    if task == "examiner" or task == "all":
        if task == "all":
            student_input_model = student_model
        if not student_input_model:
            print(
                "Missing student input model name specifies which input data used for evaluating!"
            )
            return
        llm_as_examiner(
            dataset_path,
            output_dir,
            section,
            case,
            turn,
            model_parameters={"model_name": examiner_model, "temperature": 0},
            student_input_model_name=student_input_model,
            # prompt_template="",
        )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        # required=True,
        default="all",
        help="task to run (student, examiner, or all)",
    )
    parser.add_argument(
        "-s",
        "--section",
        type=str,
        required=True,
        help="section name (qa, physical_exam, closure, diagnosis)",
    )
    parser.add_argument(
        "-c",
        "--case",
        type=str,
        default="1-10",
        help="case number",
    )
    parser.add_argument(
        "--turn",
        type=str,
        default="all",
        help="conversation turn",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="data",
        help="dataset path or directory that contain dataset",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="directory which store the output results",
    )
    parser.add_argument(
        "-sm",
        "--student_model",
        type=str,
        # default="gpt-4-1106-preview",
        help="student model name specifies which model used for generating",
    )
    parser.add_argument(
        "-sim",
        "--student_input_model",
        type=str,
        help="student input model name specifies which input data used for evaluating",
    )
    parser.add_argument(
        "-em",
        "--examiner_model",
        type=str,
        default="gpt-4-1106-preview",
        help="examiner model name specifies which model used for evaluating",
    )
    # parser.add_argument(
    #     "-p",
    #     "--prompt",
    #     type=str,
    #     help="prompt file path",
    # )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    dotenv.load_dotenv()
    args = _parse_args()
    print(args)
    main(
        args.task,
        args.section,
        args.case,
        args.turn,
        args.dataset,
        args.output,
        args.student_model,
        args.student_input_model,
        args.examiner_model,
    )

    # main(
    #     "examiner",
    #     "diagnosis",
    #     "1",
    #     "all",
    #     "data",
    #     "output",
    #     model_name="gpt-4-1106-preview",
    #     input_model_name="gpt-3.5-turbo-1106",
    # )
