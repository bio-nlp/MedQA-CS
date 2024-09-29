import os
import re
import sys
import json
import dotenv
import logging
import argparse
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Callable, Union, Tuple

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

import utils


class Section(Enum):
    qa = "qa"
    physical_exam = "physical_exam"
    closure = "closure"
    diagnosis = "diagnosis"
    other = "other"


def load_data(dataset_path: str, is_examiner: bool):
    dataset_path = Path(dataset_path).resolve()
    if dataset_path.is_dir():
        # Locate the appropriate dataset file using default naming
        if not is_examiner:
            dataset_path = dataset_path / "med-student.json"
        else:
            dataset_path = dataset_path / "med-exam.json"

    if is_examiner:
        logging.info(f"Loading examiner dataset from {dataset_path}")
    else:
        logging.info(f"Loading medical student dataset from {dataset_path}")

    with dataset_path.open("r") as f:
        dataset = json.load(f)

    return dataset


def save_result(path: str, dataset: dict, is_examiner: bool):
    output_path = Path(path).resolve()

    if output_path.suffix != "":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = output_path
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        curr_datetime = datetime.now().strftime("%m-%d-%H")
        file_prefix = "med-exam" if is_examiner else "med-student"
        output_file = output_path / f"{file_prefix}-{curr_datetime}.json"

    logging.debug(f"Saving dataset to {output_file}")
    with output_file.open("w", encoding="UTF-8") as f:
        json.dump(dataset, f, indent=2)

    return str(output_file)


def parse_range(val: str) -> Union[Tuple[int, int], None]:
    """
    Parse a string input into a range of numbers or a single number.

    Args:
        val (str): The input string to parse.

    Returns:
        Union[Tuple[int, int], None]: A tuple representing the range (start, end),
        or (number, number) for a single number, or None if the input is invalid.

    Examples:
        parse_range("1-44") -> (1, 44)
        parse_range("5") -> (5, 5)
        parse_range("") -> None
        parse_range("invalid") -> None
    """
    # Remove leading and trailing whitespace
    val = str(val).strip()

    if not val:
        logging.info("Empty range!")
        return None

    # Match a range like "1-44"
    range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", val)
    if range_match:
        start, end = map(int, range_match.groups())
        return start, end

    # Match a single number like "1"
    single_num_match = re.fullmatch(r"(\d+)", val)
    if single_num_match:
        num = int(single_num_match.group(1))
        return num, num

    logging.info("Invalid range!")
    return None


def run_model(
    model,
    prompt_template: str,
    input_data: dict[str, str],
    pre_processing_func: Callable = lambda x: x,
    post_processing_func: Callable = lambda x: x["output"],
    **kwargs,
) -> str:
    """
    Run a language model with the given prompt template and input data.

    Args:
        model: The language model to use.
        prompt_template (str): The template for generating the prompt.
        input_data (dict[str, str]): The input data to be used in the prompt.
        pre_processing (Callable, optional): Function to pre-process the input data. Defaults to identity function.
        post_processing (Callable, optional): Function to post-process the model output. Defaults to identity function.
        **kwargs: Additional keyword arguments.

    Returns:
        str: The processed result from the model.

    Process:
    1. Create a PromptTemplate from the given template.
    2. Set up a StrOutputParser for parsing the model's output.
    3. Create an evaluation chain: prompt -> model -> parser.
    4. Pre-process the input data.
    5. Run the evaluation chain with the pre-processed input.
    6. Post-process the result.
    7. Return the final processed result.
    """
    # Create prompt template and parser
    prompt = PromptTemplate.from_template(prompt_template)
    parser = StrOutputParser()

    # Set up the evaluation chain
    eval_chain = prompt | model | parser

    # Preprocess input data
    pre_processed_input = pre_processing_func(input_data)

    # Run the model and get the result
    if use_langfuse:
        config = {"callbacks": [langfuse_handler]}
    else:
        config = {}
    raw_result = eval_chain.invoke(pre_processed_input, config=config)

    logging.debug(raw_result)

    # Post-process the result
    result = post_processing_func(
        {
            "prompt": prompt_template,
            "input": input_data,
            "output": raw_result,
        }
    )

    return result


def llm_as_medical_student(
    section: str,
    case: str,
    conversation_turn: str = "all",  # only used for QA, other sections only has 1 conversation turn
    med_student_dataset_path: str = "data/med-student.json",
    output_path: str = "output/",  # the path for the output file or a path to a folder that store the output file
    model=None,  # one of the langchain model class, will override model_parameters
    model_parameters: dict = None,  # only used if model is not None
    prompt_template: dict[
        int, str
    ] = None,  # Custom prompt templates for each case number
    input_data: dict[
        int, dict[str, str]
    ] = None,  # Custom input data for each case number
    pre_processing: Callable = None,
    post_processing: Callable = None,
    **kwargs,
):
    """
    Simulates a medical student using a language model to generate responses for different sections of a medical examination.

    Args:
        section (str): The section of the medical examination (e.g., 'qa', 'physical_exam', 'closure', 'diagnosis').
        case (str): The case number or range to process.
        conversation_turn (str): The specific turn in the conversation or 'all' for all turns.
        med_student_dataset_path (str): Path to the dataset file or directory.
        output_path (str): Path to save the output file or directory.
        model: The language model to use (default is None, which will use a default model).
        model_parameters (dict): Parameters for the language model.
        prompt_template (dict): Custom prompt templates for each case.
            Example:
            {
                1: "Prompt for case 1",
                2: "Prompt for case 2"
            }
        input_data (dict): Custom input data for each case.
            Example:
            {
                1: {
                    "input_var1": "value1",
                    "input_var2": "value2"
                },
                2: {
                    "input_var1": "value3",
                    "input_var2": "value4"
                }
            }
        pre_processing (Callable): Function to preprocess input data.
        post_processing (Callable): Function to post-process model output.
        **kwargs: Additional keyword arguments.

    Returns:
        None: Results are saved to the specified output path.
    """
    logging.info(f"Running llm as medical student on {section}:")

    # Load the dataset
    dataset = load_data(med_student_dataset_path, is_examiner=False)

    # Parse case range
    start_case, end_case = (1, 44) if str(case) == "all" else parse_range(case)

    # Determine whether to use dataset prompt template or custom prompt
    use_dataset_prompt_template = prompt_template is None

    # Determine whether to use dataset input data or custom input data
    use_dataset_input_data = input_data is None

    # Set up the language model
    if model is None:
        if model_parameters is None:
            logging.info(
                "Using default model parameters: model_name=gpt-4o-mini, temperature=0.9"
            )
            model = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0.9)
        else:
            model = ChatOpenAI(**model_parameters)

    # Set default pre-processing and post-processing functions if not provided
    if pre_processing is None:
        pre_processing = lambda x: x

    # Define post-processing functions for different sections
    post_processing_func = {
        Section.qa.value: utils.medical_student_qa_post_processing,
        Section.closure.value: utils.output_only_post_processing,
        Section.physical_exam.value: utils.medical_student_physical_exam_post_processing,
        Section.diagnosis.value: utils.medical_student_diagnosis_post_processing,  # TODO: handle gpt3 broken json
    }
    if post_processing is None:
        post_processing = post_processing_func[section]

    for data in dataset:
        if (
            data["section"] == section
            and start_case <= int(data["case_id"]) <= end_case
        ):
            # Handle specific conversation turns for QA section
            if section == Section.qa.value and str(conversation_turn) != "all":
                start_conversation_turn, end_conversation_turn = parse_range(
                    conversation_turn
                )
                if not (
                    start_conversation_turn
                    <= int(data["conversation_turn_id"])
                    <= end_conversation_turn
                ):
                    continue

            logging.info(
                f'Running {data["section"]} case {data["case_id"]}, turn {data["conversation_turn_id"]}'
            )

            if use_dataset_prompt_template:
                prompt = data["prompt"]["template"]
            else:
                prompt = prompt_template[int(data["case_id"])]

            if use_dataset_input_data:
                input_data_dict = data["input"]
            else:
                input_data_dict = input_data[int(data["case_id"])]

            # Run the model
            result = run_model(
                model=model,
                prompt_template=prompt,
                input_data=input_data_dict,
                pre_processing_func=pre_processing,
                post_processing_func=post_processing,
                **kwargs,
            )

            logging.debug(result)

            # save result
            data["output"][
                model.model_name
                # "test-model"  # TODO: fix model name
            ] = result  # TODO: dataset output model name

            # save updated dataset
            output_file_path = save_result(output_path, dataset, is_examiner=False)

    logging.info(f"Finished. Output saved to: {output_file_path}")
    return output_file_path


def llm_as_examiner(
    section: str,  # one of the section in Section enum
    case: str,  # case number from 1 to 44 or "all"
    conversation_turn: str = "all",  # only used for QA, other sections only has 1 conversation turn
    med_student_dataset_path: str = None,
    med_exam_dataset_path: str = "data/med-exam.json",
    output_path: str = "output/",
    model=None,  # one of the langchain model class
    model_parameters=None,  # parameters used to initialize the model via langchainChatOpenAI class, only used if model is not None
    input_student_model_name: str = None,  # the model name of the medical student's output that will be used as the examiner's input
    prompt_template: dict[
        int, str
    ] = None,  # Custom prompt templates for each case number
    input_data: dict[
        int, dict[str, str]
    ] = None,  # Custom input data for each case number
    pre_processing: Callable = None,  # function to preprocess input data
    post_processing: Callable = None,  # function to post-process model output
    **kwargs,
):
    """
    Simulates a medical examiner using a language model to evaluate responses from medical students.

    Args:
        section (str): The section of the medical examination (e.g., 'qa', 'physical_exam', 'closure', 'diagnosis').
        case (str): The case number or range to process.
        conversation_turn (str): The specific turn in the conversation or 'all' for all turns.
        med_student_dataset_path (str): Path to the dataset file or directory.
        med_exam_dataset_path (str): Path to the dataset file or directory.
        output_path (str): Path to save the output file or directory.
        model: The language model to use (default is None, which will use a default model).
        model_parameters (dict): Parameters for the language model.
        input_student_model_name (str): The model name of the medical student's output that will be used as the examiner's input.
        prompt_template (dict): Custom prompt templates for each case.
            Example:
            {
                1: "Prompt for case 1",
                2: "Prompt for case 2"
            }
        input_data (dict): Custom input data for each case.
            Example:
            {
                1: {
                    "input_var1": "value1",
                    "input_var2": "value2"
                },
                2: {
                    "input_var1": "value3",
                    "input_var2": "value4"
                }
            }
        pre_processing (Callable): Function to preprocess input data.
        post_processing (Callable): Function to post-process model output.
        **kwargs: Additional keyword arguments.

    Returns:
        None: Results are saved to the specified output path.
    """
    logging.info(f"Running llm as examiner on {section}:")

    # Check if input_student_model_name is provided
    if input_student_model_name is None:
        logging.error(
            "Missing student model name specified which input data used to evaluate!"
        )
        sys.exit(1)

    # Check if med_exam_dataset_path is provided
    if med_exam_dataset_path is None:
        logging.error("Need specify med-exam dataset!")
        sys.exit(1)
    # Load medical student dataset if provided
    if med_student_dataset_path:
        logging.info("Use med-student and med-exam dataset")
        student_dataset = load_data(med_student_dataset_path, is_examiner=False)
    else:
        logging.info("Use med-exam dataset only")
        student_dataset = None

    # Load medical examiner dataset
    dataset = load_data(med_exam_dataset_path, is_examiner=True)

    start_case, end_case = (1, 44) if str(case) == "all" else parse_range(case)
    # TODO: Stop running if parse return None

    # Determine whether to use dataset prompt template or custom prompt
    use_dataset_prompt_template = prompt_template is None

    # Determine whether to use dataset input data or custom input data
    use_dataset_input_data = input_data is None

    if model is None:
        if model_parameters is None:
            logging.info(
                "Using default model parameters: model_name=gpt-4-1106-preview, temperature=0"
            )
            model = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
        else:
            model = ChatOpenAI(**model_parameters)

    # Set default pre-processing and post-processing functions if not provided
    if pre_processing is None:
        pre_processing = lambda x: x

    # Define post-processing functions for different sections
    post_processing_func = {
        Section.qa.value: utils.output_only_post_processing,
        Section.physical_exam.value: utils.output_only_post_processing,
        Section.closure.value: utils.output_only_post_processing,
        Section.diagnosis.value: utils.examiner_diagnosis_post_processing,
    }
    if post_processing is None:
        post_processing = post_processing_func[section]

    # get examiner model name
    examiner_model_name = model.model_name

    for index, data in enumerate(dataset):
        if (
            data["section"] == section
            and start_case <= int(data["case_id"]) <= end_case
        ):
            # Handle specific conversation turns for QA section
            if section == Section.qa.value and str(conversation_turn) != "all":
                start_conversation_turn, end_conversation_turn = parse_range(
                    conversation_turn
                )
                if not (
                    start_conversation_turn
                    <= int(data["conversation_turn_id"])
                    <= end_conversation_turn
                ):
                    continue

            logging.info(
                f'Running {data["section"]} case {data["case_id"]}, turn {data["conversation_turn_id"]}'
            )

            if use_dataset_prompt_template:
                prompt = data["prompt"]["template"]
            else:
                prompt = prompt_template[int(data["case_id"])]

            if use_dataset_input_data:
                # input_data_dict = data["input"]
                if section == Section.qa.value:
                    input_dict_name = "question"
                else:
                    input_dict_name = "pred"

                # choose one of the student model's output as examiner input
                if input_student_model_name not in data["input"][input_dict_name]:
                    # find input data from med-student dataset
                    if student_dataset is None:
                        logging.error(
                            f"Cannot find input data for model {input_student_model_name}!!!"
                        )
                        logging.error(
                            "Consider including the med-student dataset in the examiner task."
                        )
                        sys.exit(1)
                    student_data = student_dataset[index]
                    if (
                        student_data["section"] != section
                        or student_data["case_id"] != data["case_id"]
                        or student_data["conversation_turn_id"]
                        != data["conversation_turn_id"]
                    ):
                        logging.error("Error, student dataset info don't match!!!")
                        # TODO: student dataset full search
                        sys.exit(1)

                    # add med-student output to med-exam dataset as examiner input
                    if input_student_model_name in student_data["output"]:
                        data["input"][input_dict_name][input_student_model_name] = (
                            student_data["output"][input_student_model_name]
                        )
                    else:
                        logging.error(
                            f"Cannot find input data for model {input_student_model_name}!!!"
                        )
                        logging.error(
                            "Consider including the med-student dataset in the examiner task."
                        )
                        sys.exit(1)

                # setting examiner model input data
                input_data_dict = {}
                for key, value in data["input"].items():
                    if key == input_dict_name:
                        input_data_dict[key] = value[input_student_model_name]
                    else:
                        input_data_dict[key] = value

            else:
                input_data_dict = input_data[int(data["case_id"])]

            result = run_model(
                model=model,
                prompt_template=prompt,
                input_data=input_data_dict,
                pre_processing_func=pre_processing,
                post_processing_func=post_processing,
                **kwargs,
            )

            logging.debug(result)

            # save result
            if input_student_model_name not in data["output"]:
                data["output"][input_student_model_name] = {}
                logging.debug(f"Creating new entry for {input_student_model_name}")
            data["output"][input_student_model_name][examiner_model_name] = result

            # save updated dataset
            output_file_path = save_result(output_path, dataset, is_examiner=True)

    logging.info(f"Finished. Output saved to: {output_file_path}")
    return output_file_path


def main(args):
    """
    Main function to run the medical examination simulation.

    This function handles the execution of tasks for both the medical student and examiner roles,
    based on the provided command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing task specifications
                                   and other parameters.

    Returns:
        None

    Raises:
        None, but exits the function early if required arguments are missing.
    """

    if args.task not in ["student", "examiner", "all"]:
        logging.error("Invalid task! Please choose 'student', 'examiner', or 'all'.")
        sys.exit(1)

    if args.task in ["student", "all"]:
        if not args.student_model:
            logging.error(
                "Missing student model name. Please specify which model to use for generating responses."
            )
            sys.exit(1)
        if not args.med_student_dataset:
            logging.error(
                "Missing medical student dataset. Please specify which dataset to use for generation."
            )
            sys.exit(1)

    if args.task in ["examiner", "all"]:
        if not args.student_model:
            logging.error(
                "Missing student model name. Please specify which input data to use for evaluation."
            )  # TODO: improve wording
            sys.exit(1)
        if not args.examiner_model:
            logging.error(
                "Missing examiner model name. Please specify which model to use for evaluation."
            )  # TODO: improve wording
            sys.exit(1)
        # if not args.med_student_dataset:# TODO:add documentation
        #     logging.error(
        #         "Missing medical student dataset. Please specify the dataset containing the student model's output to be used as input for evaluation."
        #     )
        #     sys.exit(1)
        if not args.med_exam_dataset:
            logging.error(
                "Missing medical examination dataset. Please specify which dataset to use for evaluation."
            )
            sys.exit(1)

    logging.info(f"Starting task: {args.task}")
    logging.info(f"Section: {args.section}, Case: {args.case}, Turn: {args.turn}")
    if args.task == "student":
        logging.info(f"Model: {args.student_model}")
    else:
        logging.info(f"Model: {args.student_model}, {args.examiner_model}")

    try:
        if args.task in ["student", "all"]:
            med_student_output_file_path = llm_as_medical_student(
                section=args.section,
                case=args.case,
                conversation_turn=args.turn,
                med_student_dataset_path=args.med_student_dataset,
                output_path=args.output,
                model_parameters={"model_name": args.student_model, "temperature": 0.9},
                # prompt_template="",  # Uncomment and provide a template if needed
            )

        if args.task in ["examiner", "all"]:
            if args.task == "all":
                # Use the path of the dataset generated by llm_as_medical_student
                med_student_dataset_path = med_student_output_file_path
            else:
                if args.med_student_dataset == "data/med-student.json":
                    med_student_dataset_path = None
                else:
                    med_student_dataset_path = args.med_student_dataset
            llm_as_examiner(
                section=args.section,
                case=args.case,
                conversation_turn=args.turn,
                med_student_dataset_path=med_student_dataset_path,
                med_exam_dataset_path=args.med_exam_dataset,
                output_path=args.output,
                model_parameters={"model_name": args.examiner_model, "temperature": 0},
                input_student_model_name=args.student_model,
                # prompt_template="",  # Uncomment and provide a template if needed
            )

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        logging.error("Use verbose mode to see detailed error information.")
        if args.verbose:
            logging.exception("Detailed error information:")
        sys.exit(1)

    logging.info("Task completed successfully.")


def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    log_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

    # Set up logging to file
    logging.basicConfig(
        level=level, format=log_format, filename="MedQA-CS.log", filemode="a"
    )

    # Set up logging to console
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter(log_format)
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def parse_args():
    """
    Parse command-line arguments for the medical examination simulation.

    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Medical Examination Simulation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task selection
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        required=True,
        default="all",
        choices=["student", "examiner", "all"],
        help="Task to run: student (generate responses), examiner (evaluate responses), or all (both)",
    )
    parser.add_argument(
        "-s",
        "--section",
        type=str,
        required=True,
        choices=["qa", "physical_exam", "closure", "diagnosis"],
        help="Section of the medical examination (qa, physical_exam, closure, diagnosis)",
    )
    parser.add_argument(
        "-c",
        "--case",
        type=str,
        required=True,
        help="Case number or range (e.g., '1-44' for cases 1 through 44)",
    )
    parser.add_argument(
        "--turn",
        type=str,
        default="all",
        help="Specific conversation turn or 'all' for entire conversation",
    )
    parser.add_argument(
        "-sd",
        "--med_student_dataset",
        type=str,
        default="data/med-student.json",
        help="Path to the medical student dataset for generation task",
    )
    parser.add_argument(
        "-ed",
        "--med_exam_dataset",
        type=str,
        default="data/med-exam.json",
        help="Path to the medical examination dataset for examiner task",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output/",
        help="Path to output file or directory. If a directory is specified, output files will be saved with default names.",
    )
    parser.add_argument(
        "-sm",
        "--student_model",
        type=str,
        # default="gpt-4o-mini",
        help="Name of the model to use for generating student responses",
    )
    parser.add_argument(
        "-em",
        "--examiner_model",
        type=str,
        default="gpt-4-1106-preview",
        help="Name of the model to use for evaluating responses",
    )
    # parser.add_argument(
    #     "-p",
    #     "--prompt",
    #     type=str,
    #     help="Path to the prompt file",
    # )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()
    return args


def setup_langfuse():
    if (
        os.environ.get("LANGFUSE_PUBLIC_KEY")
        and os.environ.get("LANGFUSE_SECRET_KEY")
        and os.environ.get("LANGFUSE_HOST")
    ):
        from langfuse.callback import CallbackHandler

        logging.info("Using LangFuse")
        return True, CallbackHandler()
    return False, None


if __name__ == "__main__":
    args = parse_args()
    dotenv.load_dotenv()
    setup_logging(args.verbose)
    logging.info(args)
    use_langfuse, langfuse_handler = setup_langfuse()

    with get_openai_callback() as cb:
        main(args)

        logging.info("-" * 40)
        logging.info(f"Successful Requests: {cb.successful_requests}")
        logging.info(f"Total Tokens: {cb.total_tokens}")
        logging.info(f"Prompt Tokens: {cb.prompt_tokens}")
        logging.info(f"Completion Tokens: {cb.completion_tokens}")
        logging.info(f"Total Cost (USD): ${cb.total_cost}")
        logging.info("-" * 40)
