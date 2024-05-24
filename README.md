# MedQA-CS
Benchmarking LLMs Clinical Skills for Patient-Centered Diagnostics and Documentation

# How to run

`run_dataset.py` is designed to run a Language Model (LLM) on a JSON dataset for medical student and examiner tasks. The program supports several sections, including Question & Answer (QA), Physical Exam, Closure, and Diagnosis.

## Prerequisites
```
pip install langchain,langchain-openai,python-dotenv
```

- Get an OpenAI api key and set it as an environment variable (`OPENAI_API_KEY`)

## Run with Command Line Arguments

To run the program, use the following command:

```
python run_dataset.py --task TASK --section SECTION [--case CASE] [--turn TURN] [--dataset DATASET_PATH] [--output OUTPUT_DIR] [-m MODEL] [--student_model STUDENT_MODEL]
```

### Arguments

- `--task` (Required): Specify the task to run. Options are `student` or `examiner`.
- `--section` (Required): Specify the section name from the dataset to process. Options are `qa`, `physical_exam`, `closure`, or `diagnosis`.
- `--case` (default: `1-10`): Specify the case number or range of cases to run. Use a hyphen (-) to specify a range. (e.g. `5-12`)
- `--turn` (default: `all`): Specify the conversation turn or range of turns to run. Use a hyphen (-) to specify a range.  Use `all` to run all turns. (e.g. `1-5`)
- `--dataset` (default: `dataset`): Specify the path to the dataset directory or the dataset JSON file.
- `--output` (default: `output`): Specify the directory where the output results will be stored.
- `-m`, `--model` (default: `gpt-4`): Specify the model name to use for the LLM.
- `--student_model` (optional): Specify the input student model name used for evaluation when running the `examiner` task.

## Examples

1. Run LLM as `student` task for the `qa` section on cases 1-10 and all conversation turns, using the `gpt-4-1106-preview` model and the dataset located in the `dataset` directory:

```
python run_dataset.py --task student --section qa --case 1-10 --turn all --dataset dataset --output output -m gpt-4-1106-preview
```

2. Run the `examiner` task for the `physical_exam` section on case 5, using the `gpt-4-1106-preview` model as examiner and the evaluating the input of student's answer from the `gpt-3.5-turbo-1106` model, dataset located in the `dataset` directory, output in `/output` directory, :

```
python run_dataset.py --task examiner --section physical_exam --case 5 --dataset dataset --output output -m gpt-4-1106-preview --input_model gpt-3.5-turbo-1106
```

## Run from program


## Function and Input Argument Explanations
- `load_data(dataset_path, generation)`: Loads data from a JSON file and organizes it into a lookup structure. It takes a path to the dataset and a boolean indicating whether to load the generation or evaluation dataset.

- `save_result(path, result)`: Saves the given result string to a file at the specified path.

- `parse_range(val)`: Parses a string that may represent a range (e.g., "1-10") or a single number (e.g., "5") and returns a tuple of integers.

- `run_model(llm, prompt_template, input_data, prev_processing, post_processing, **kwargs)`: Executes the LLM with the given prompt template and input data. It includes optional pre- and post-processing functions.

- `llm_as_medical_student(*args, **kwargs)`: Simulates an LLM acting as a medical student on a dataset. It accepts various arguments for customization.

- `llm_as_examiner(*args, **kwargs)`: Simulates an LLM acting as an examiner on a dataset. It also accepts various arguments for customization.

- `main(*args, **kwargs)`: The main function that orchestrates the execution of the program based on the provided command-line arguments.

- `_parse_args()`: Utilizes argparse to define and parse command-line arguments for the program.

## Notes
- Make sure you have the required dependencies installed (`langchain`, `langchain_openai`, `python-dotenv`) before running the program.
- The langchain library is used for interacting with the LLM. Ensure you have the necessary permissions and API keys if required by the library.

## Todo
- ~~Add `all` task~~
- ~~Store result to new dataset~~
- Store medical student result to both dataset
- Add `prompt_path`
- ~~Add `student_model`, `student_input_model`, `examiner_model`~~
- Add model name similar matching
- ~~Add data viewer~~
- Improve data viewer
- Add logging

# Dataset Format

## `generation.json`

Each entry in the JSON file is structured as follows:

- **`unique_id`**: A unique identifier to distinguish each entry.

- `section`: The type of content, which can be one of the following:
    - `qa`: Question-answer section.
    - `physical_exam`: Physical examination section.
    - `closure`: Closure section.
    - `diagnosis`: Diagnosis section.

- `case_id`: The identifier of the case number, ranging from 1 to 44.

- `conversation_turn_id`: Indicates the sequence of dialogue for the `qa` section. All other sections have `conversation_turn_id` with 1.

- `input`: An object containing the input variable name and input data.

- `ground_truth_output`: The expected response or output from the examiner.

- `prompt`: An object containing the prompt:
    - `_type`: A string indicate this a prompt
    - `input_variables`: An array of input variable names used in the prompt template.
    - `template`: A prompt template that includes placeholders for the input variables.

- `output`: An object containing the running model name and output data generate from that model.

### Example Case Object
```
{
  "unique_id": 1,
  "section": "qa",
  "case_id": 1,
  "conversation_turn_id": 1,
  "input": {
    "opening": "Opening Scenario:\n\...
    "chat_history": "N/A"
  },
  "ground_truth_output": "\"What con...
  "prompt": {
    "_type": "prompt",
    "input_variables": [
      "opening",
      "chat_history"
    ],
    "template": "You are a doctor an...
  },
  "output": {
    "gpt-3.5-turbo-1106": "When did ...
    "gpt-4-1106-preview": "Can you d...
    "claude-3-sonnet-20240229": "Can...
    "claude-3-haiku-20240307": "Can ...
    "claude-3-opus-20240229": "What ...
  }
```


## `evaluation.json`
Each entry in the JSON file is structured as follows:

- **`unique_id`**: A unique identifier to distinguish each entry.

- `section`: The type of content, which can be one of the following:
    - `qa`: Question-answer section.
    - `physical_exam`: Physical examination section.
    - `closure`: Closure section.
    - `diagnosis`: Diagnosis section.

- `case_id`: The identifier of the case number, ranging from 1 to 44.

- `conversation_turn_id`: Indicates the sequence of dialogue for the `qa` section. All other sections have `conversation_turn_id` with 1.

- `input`: An object containing the model name that generate this input data waiting for evaluating, and another object for each input model name containing the input variable name and input data.

- `prompt`: An object containing the prompt:
    - `_type`: A string indicate this a prompt
    - `input_variables`: An array of input variable names used in the prompt template.
    - `template`: A prompt template that includes placeholders for the input variables.

- `result`: An object containing the model name for the input data and evaluation result for that input model name.

### Example Case Object
```
{
  "unique_id": 40,
  "section": "diagnosis",
  "case_id": 1,
  "conversation_turn_id": 1,
  "input": {
    "gpt-3.5-turbo-1106": {
      "pred": "Diagnosis #1: Acute Coronary...
      "target": "Diagnosis #1: Myocardial i...
      "additional_diagnosis": "-Aortic diss...
    },
    "gpt-4-1106-preview": {
      "pred": "Diagnosis #1: Acute Coronary...
      "target": "Diagnosis #1: Myocardial i...
      "additional_diagnosis": "-Aortic diss...
    },
    "claude-3-sonnet-20240229": {
      "pred": "Diagnosis #1: Acute Coronary...
      "target": "Diagnosis #1: Myocardial i...
      "additional_diagnosis": "-Aortic diss...
    },
    "claude-3-haiku-20240307": {
      "pred": "",
      "target": "Diagnosis #1: Myocardial i...
      "additional_diagnosis": "-Aortic diss...
    },
    "claude-3-opus-20240229": {
      "pred": "",
      "target": "Diagnosis #1: Myocardial i...
      "additional_diagnosis": "-Aortic diss...
    }
  },
  "prompt": {
    "_type": "prompt",
    "input_variables": [
      "predicted_diagnosis",
      "target_diagnosis",
      "additional_diagnosis"
    ],
    "template": "You are an evaluator for t...
  },
  "result": {
    "gpt-3.5-turbo-1106": "{\n  \"diagnosis...
    "gpt-4-1106-preview": "{\n  \"diagnosis...
    "claude-3-sonnet-20240229": "{\n  \"dia...
    "claude-3-haiku-20240307": "",
    "claude-3-opus-20240229": ""
```