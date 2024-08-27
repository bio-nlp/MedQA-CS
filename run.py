import os
import re
import json
import dotenv
import argparse
from enum import Enum
from typing import Callable
from datetime import datetime
from collections import defaultdict

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from datasets import load_dataset

import utils


class Section(Enum):
    qa = "qa"
    physical_exam = "physical_exam"
    closure = "closure"
    diagnosis = "diagnosis"
    other = "other"


# Function to load the dataset from a given path
# def load_data(dataset_path, examiner):
#     print(dataset_path)
#     if os.path.isdir(dataset_path):
#         # Determine the correct dataset file based on the generation flag
#         if not examiner:
#             dataset_path = os.path.join(dataset_path, "med-student.json")
#         else:
#             dataset_path = os.path.join(dataset_path, "med-exam.json")
#
#     print(f"Loading data from {dataset_path}")
#     with open(dataset_path, "r") as f:
#         data = json.load(f)
#
#     # Create a lookup dictionary to map sections and case IDs to conversation turns
#     lookup = dict(
#         zip(
#             [section.value for section in Section],
#             [defaultdict(dict) for i in range(5)],
#         )
#     )
#
#     prev_section = Section.qa.value
#     prev_case = 1
#     max_turn = 0
#     for i, element in enumerate(data):
#         section = element["section"]
#         case_id = int(element["case_id"])
#         conversation_turn_id = int(element["conversation_turn_id"])
#         lookup[section][case_id][conversation_turn_id] = i
#
#         if prev_section != section or prev_case != case_id:
#             lookup[prev_section][prev_case]["max_turn"] = max_turn
#             prev_section = section
#             prev_case = case_id
#             max_turn = 0
#         max_turn += 1
#
#     lookup[prev_section][prev_case]["max_turn"] = max_turn
#
#     return data, lookup


# Function to load the dataset from a given path or folder
def load_data(dataset_path, is_examiner):
    # if dataset_path is None:
    #     if not is_examiner:
    #         print(f"Loading dataset from huggingface bio-nlp-umass/MedQA-CS-Student")
    #         dataset = load_dataset("bio-nlp-umass/MedQA-CS-Student")
    #         dataset["train"].to_json("data/med-student.json", lines=False, indent=2)
    #     else:
    #         print(f"Loading dataset from huggingface bio-nlp-umass/MedQA-CS-Exam")
    #         dataset = load_dataset("bio-nlp-umass/MedQA-CS-Exam")
    #         dataset["train"].to_json("data/med-exam.json", lines=False, indent=2)

    if os.path.isdir(dataset_path):
        # Find correct dataset file with default name
        if not is_examiner:
            dataset_path = os.path.join(dataset_path, "med-student.json")
        else:
            dataset_path = os.path.join(dataset_path, "med-exam.json")
    print(f"Loading dataset from {dataset_path}")
    # dataset = load_dataset("json", data_files=dataset_path)
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    return dataset


def save_result(path: str, dataset: dict):
    print(f"save dataset to {path}")
    with open(path, "w", encoding="UTF-8") as f:
        json.dump(dataset, f, indent=2)


# Function to parse a range or a single number from a string
def parse_range(val: str):
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
    model,
    prompt_template: str,
    input_data: dict[str:str],
    pre_processing: Callable = lambda x: x,
    post_processing: Callable = lambda x: x,
    **kwargs,
):
    prompt = PromptTemplate.from_template(prompt_template)

    parser = StrOutputParser()

    eval_chain = prompt | model | parser

    input_data = pre_processing(input_data)

    result = eval_chain.invoke(input_data)

    result = post_processing(result)

    return result


def llm_as_medical_student(
    section: str,
    case: str,
    conversation_turn: str = "all",  # only used for QA, other sections only has 1 conversation turn
    dataset_path: str = "data/",
    output_path: str = "output/",  # the path for the output file or a path to a folder that store the output file
    model=None,  # one of the langchain model class
    model_parameters: dict = None,
    prompt_template: dict[str:str] = None,  # example: {1,"prompt",2:"prompt"}
    input_data: dict[str : dict[str:str]] = None,
    pre_processing: Callable = None,
    post_processing: Callable = None,
    **kwargs,
):
    print(f"Running llm as medical student on {section}:")

    post_processing_func = {
        Section.qa.value: lambda x: x,
        Section.closure.value: lambda x: x,
        Section.physical_exam.value: utils.medical_student_physical_exam_post_processing,
        Section.diagnosis.value: utils.medical_student_diagnosis_post_processing,  # TODO: handle gpt3 broken json
    }

    dataset = load_data(dataset_path, is_examiner=False)

    start_case, end_case = (1, 44) if str(case) == "all" else parse_range(str(case))

    if prompt_template is None:
        use_dataset_prompt_template = True  # TODO: variable name
    else:
        use_dataset_prompt_template = False

    # if input_data is None:
    #     use_ds_input_data = True
    # else:
    #     use_ds_input_data = False
    use_dataset_input_data = input_data is None

    if model is None:
        if model_parameters is None:
            model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.9)
        else:
            model = ChatOpenAI(**model_parameters)
            # TODO: handle empty model_para if llm is None
    if pre_processing is None:
        pre_processing = lambda x: x
    if post_processing is None:
        post_processing = post_processing_func[section]

    # for case in range(start_case, end_case + 1):
    #     if section == "qa":
    #         if str(conversation_turn) == "all":
    #             start_conversation_turn = 1
    #             end_conversation_turn = lookup[section][case]["max_turn"]
    #         else:
    #             start_conversation_turn, end_conversation_turn = parse_range(
    #                 str(conversation_turn)
    #             )
    #     else:
    #         start_conversation_turn = 1
    #         end_conversation_turn = 1
    #
    #     for turn in range(start_conversation_turn, end_conversation_turn + 1):
    #         pass

    for data in dataset:
        if data["section"] == section and start_case <= data["case_id"] <= end_case:
            if section == Section.qa.value and str(conversation_turn) != "all":
                start_conversation_turn, end_conversation_turn = parse_range(
                    str(conversation_turn)
                )
                if not (
                    start_conversation_turn
                    <= data["conversation_turn_id"]
                    <= end_conversation_turn
                ):
                    continue
            print(
                f'Running {data["section"]} case {data["case_id"]}, turn {data["conversation_turn_id"]}'
            )

            if use_dataset_prompt_template:
                prompt = data["prompt"]["template"]
            else:
                prompt = prompt_template[case]

            if use_dataset_input_data:
                input_data_dict = data["input"]
            else:
                input_data_dict = input_data[case]

            result = run_model(
                model=model,
                prompt_template=prompt,
                input_data=input_data_dict,
                pre_processing=pre_processing,
                post_processing=post_processing,
                **kwargs,
            )

            print(result)

            # save result
            data["output"][
                # llm.model_name
                "test-model"  # TODO: fix model name
            ] = result  # TODO: dataset output model name

            output_file = output_path
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            if os.path.isdir(output_file):
                curr_day = datetime.now().strftime("%m-%d")
                output_file = os.path.join(output_file, f"med-student-{curr_day}.json")
            save_result(output_file, dataset)

    print("Finish")


def llm_as_examiner(
    section: str,
    case: str,
    conversation_turn: str = "all",  # only used for QA, other sections only has 1 conversation turn
    dataset_path: str = "data/",
    med_student_dataset_path=None,
    med_exam_dataset_path=None,
    output_path: str = "output/",
    model=None,  # one of the langchain model class
    model_parameters=None,
    input_student_model_name: str = None,
    prompt_template: str = None,
    input_data: dict[str:str] = None,
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

    if med_student_dataset_path is None and med_exam_dataset_path is None:
        print("Use default dataset...")
        student_dataset = load_data(dataset_path, is_examiner=False)
        dataset = load_data(dataset_path, is_examiner=True)
    else:
        print("Use given dataset...")
        if med_student_dataset_path is None or med_exam_dataset_path is None:
            print("Need specify both med-student and med-exam dataset!")
            return
        student_dataset = load_data(med_student_dataset_path, is_examiner=False)
        dataset = load_data(med_exam_dataset_path, is_examiner=True)

    start_case, end_case = (1, 44) if str(case) == "all" else parse_range(str(case))
    # TODO: Stop running if parse return None

    if prompt_template is None:
        use_dataset_prompt_template = True
    else:
        use_dataset_prompt_template = False

    if input_data is None:
        if input_student_model_name is None:
            print(
                "Missing student model name specified which input data used to evaluate!"
            )
            return
        use_dataset_input_data = True
    else:
        use_dataset_input_data = False

    if model is None:
        if model_parameters is None:
            model = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
        else:
            model = ChatOpenAI(**model_parameters)
            # TODO: handle empty model_param if model is None
    if pre_processing is None:
        pre_processing = lambda x: x
    if post_processing is None:
        post_processing = post_processing_func[section]

    # for case in range(start_case, end_case + 1):
    #     if str(conversation_turn) == "all":
    #         start_conversation_turn = 1
    #         end_conversation_turn = lookup[section][case]["max_turn"]
    #     else:
    #         start_conversation_turn, end_conversation_turn = parse_range(
    #             str(conversation_turn)
    #         )
    #
    #     for turn in range(start_conversation_turn, end_conversation_turn + 1):
    #         print(f"Running case {case}, turn {turn}")
    #
    #         index = lookup[section][case][turn]
    #         data = dataset[index]

    for index, data in enumerate(dataset):
        if data["section"] == section and start_case <= data["case_id"] <= end_case:
            if section == Section.qa.value and str(conversation_turn) != "all":
                start_conversation_turn, end_conversation_turn = parse_range(
                    str(conversation_turn)
                )
                if not (
                    start_conversation_turn
                    <= data["conversation_turn_id"]
                    <= end_conversation_turn
                ):
                    continue
            print(
                f'Running {data["section"]} case {data["case_id"]}, turn {data["conversation_turn_id"]}'
            )

            if use_dataset_prompt_template:
                prompt = data["prompt"]["template"]
            else:
                prompt = prompt_template[case]

            if use_dataset_input_data:
                # input_data_dict = data["input"]

                if section == Section.qa.value:
                    input_dict_name = "question"
                else:
                    input_dict_name = "pred"

                # choose one of the student model's output as examiner input
                if input_student_model_name not in data["input"][input_dict_name]:
                    # find input data from med-student dataset
                    student_data = student_dataset[index]
                    if (
                        student_data["section"] != section
                        or student_data["case_id"] != data["case_id"]
                        or student_data["conversation_turn_id"]
                        != data["conversation_turn_id"]
                    ):
                        print("Error, student dataset info don't match!!!")
                        # TODO: student dataset full search

                    if input_student_model_name in student_data["output"]:
                        data["input"][input_dict_name][input_student_model_name] = (
                            student_data["output"][input_student_model_name]
                        )
                    else:
                        print(
                            f"Cannot find input data for model {input_student_model_name}!!!"
                        )
                        return

                # input_data_dict[input_dict_name] = data["input"][input_dict_name][
                #     input_student_model_name
                # ]
                input_data_dict = {}
                for key, value in data["input"].items():
                    if key == input_dict_name:
                        input_data_dict[key] = value[input_student_model_name]
                    else:
                        input_data_dict[key] = value

            else:
                input_data_dict = input_data[case]

            result = run_model(
                model=model,
                prompt_template=prompt,
                input_data=input_data_dict,
                pre_processing=pre_processing,
                post_processing=post_processing,
                **kwargs,
            )

            print(result)

            # save result
            # data["output"][llm.model_name] = result #TODO: fix model name
            data["output"]["test-model"] = result

            output_file = output_path
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            if os.path.isdir(output_file):
                curr_day = datetime.now().strftime("%m-%d")
                output_file = os.path.join(output_file, f"med-exam-{curr_day}.json")
            save_result(output_file, dataset)

    print("Finish")


def main(
    args,
    # task,
    # section,
    # case,
    # turn,
    # dataset_path,
    # output_dir,
    # student_model=None,
    # # student_input_model=None,
    # examiner_model=None,
):
    if args.task != "student" and args.task != "examiner" and args.task != "all":
        print("Invalid task!")
        return

    if args.task == "student" or args.task == "all":
        if not args.student_model:
            print(
                "Missing student model name specifies which model used for generating!"
            )
            return
        llm_as_medical_student(
            section=args.section,
            case=args.case,
            turn=args.turn,
            dataset_path=args.dataset,
            output_path=args.output,
            model_parameters={"model_name": args.student_model, "temperature": 0.9},
            # prompt_template="",
        )
    if args.task == "examiner" or args.task == "all":
        # if task == "all":
        #     student_input_model = student_model
        if args.student_model is None:
            print(
                # TODO: Improve wording
                "Missing student model name specifies which input data used for evaluating!"
            )
            return
        llm_as_examiner(
            section=args.section,
            case=args.case,
            turn=args.turn,
            dataset_path=args.dataset,
            med_student_dataset_path=args.med_student_dataset,
            med_exam_dataset_path=args.med_exam_dataset,
            output_path=args.output,
            model_parameters={"model_name": args.examiner_model, "temperature": 0},
            input_student_model_name=args.student_model,
            # prompt_template="",
        )


def parse_args():
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
        default="data/",
        help="dataset path or directory that contain dataset",
    )
    parser.add_argument(
        "--med_student_dataset",
        type=str,
        help="med-student dataset path used for examiner task",
    )
    parser.add_argument(
        "--med_exam_dataset",
        type=str,
        help="med-exam dataset path used for examiner task",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output/",
        help="output dataset path or directory that store output dataset with default name",
    )
    parser.add_argument(
        "-sm",
        "--student_model",
        type=str,
        # default="gpt-3.5-turbo-1106",
        help="student model name specifies which model used for generating",
    )
    # parser.add_argument(
    #     "-sim",
    #     "--student_input_model",
    #     type=str,
    #     default="gpt-4-1106-preview",
    #     help="student input model name specifies which input data used for evaluating",
    # )
    parser.add_argument(
        "-em",
        "--examiner_model",
        type=str,
        # default="gpt-4-1106-preview",
        help="examiner model name specifies which model used for evaluating",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="prompt file path",
    )
    args = parser.parse_args()
    return args


def test(args):
    print(args.med_student_dataset)
    print(type(args.med_student_dataset))
    if args.med_student_dataset:
        print("true")
    print(args.aaa)
    quit()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    dotenv.load_dotenv()
    # test(args)
    main(
        args
        # args.task,
        # args.section,
        # args.case,
        # args.turn,
        # args.dataset,
        # args.output,
        # args.student_model,
        # # args.student_input_model,
        # args.examiner_model,
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
