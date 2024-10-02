# MedQA-CS
Benchmarking LLMs Clinical Skills for Patient-Centered Diagnostics and Documentation

## Dataset
[MedQA-CS-Student](https://huggingface.co/datasets/bio-nlp-umass/MedQA-CS-Student) and [MedQA-CS-Exam](https://huggingface.co/datasets/bio-nlp-umass/MedQA-CS-Exam) are available through Huggingface.

**<span style="color: red; font-weight: bold;">⚠️ Important: Please note that the scores currently obtained using the GPT-4 judge may differ from those obtained a few months ago. We are aware of this discrepancy and are working on updating to a Llama-examiner to address this issue. Please keep mindful when using the data for benchmarking or comparison.</span>**

## How to run

`main.py` is designed to run a Language Model (LLM) on JSON datasets for medical student and examiner tasks. The program supports several sections, including Question & Answer (QA), Physical Exam, Closure, and Diagnosis.

### Installation
We used `python 3.10` to develop this project.
```
pip install -r requirements.txt
```

#### Run LLM with OpenAI API
Create a environment variable file `.env` in the root directory and set your OpenAI API key in it.
```
OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
```

### Run with Command Line Arguments

To run the program, use the following command:

```
python main.py [-h] -t {student,examiner,all} -s {qa,physical_exam,closure,diagnosis} -c CASE [--turn TURN] [-sd MED_STUDENT_DATASET] [-ed MED_EXAM_DATASET] [-o OUTPUT] [-sm STUDENT_MODEL] [-em EXAMINER_MODEL] [-v]
```

#### Arguments
```
  -h, --help            show this help message and exit
  -t {student,examiner,all}, --task {student,examiner,all}
                        Task to run: student (generate responses), examiner (evaluate responses), or all (both)
  -s {qa,physical_exam,closure,diagnosis}, --section {qa,physical_exam,closure,diagnosis}
                        Section of the medical examination
  -c CASE, --case CASE  Single case number or a range of case numbers or 'all' (e.g., '1-44' for cases 1 through 44)
  --turn TURN           Specific conversation turn or 'all' for entire conversation (default: all)
  -sd MED_STUDENT_DATASET, --med_student_dataset MED_STUDENT_DATASET
                        Path to the medical student dataset for generation task (default: data/med-student.json)
  -ed MED_EXAM_DATASET, --med_exam_dataset MED_EXAM_DATASET
                        Path to the medical examination dataset for examiner task (default: data/med-exam.json)
  -o OUTPUT, --output OUTPUT
                        Path to output file or directory. If a directory is specified, output files will be saved
                        with default names. (default: output/)
  -sm STUDENT_MODEL, --student_model STUDENT_MODEL
                        Name of the model to use for generating student responses
  -em EXAMINER_MODEL, --examiner_model EXAMINER_MODEL
                        Name of the model to use for evaluating responses (default: gpt-4-1106-preview)
  -v, --verbose         Enable verbose output
```

### Examples

1. Run LLM as `student` task for the `qa` section on cases 1-10 and all conversation turns, using the `gpt-4o-mini` model:

```
python main.py --task student --section qa --case 1-10 --turn all --med_student_dataset ./data/med-student.json --output ./output --student_model gpt-4o-mini
```

2. Run the `examiner` task for the `physical_exam` section on case 5, using the `gpt-4-1106-preview` model as examiner and evaluating the input of student's answer from the `gpt-3.5-turbo-1106` model:

```
python main.py --task examiner --section physical_exam --case 5 --med_exam_dataset ./data/med-exam.json --output ./output --student_model gpt-3.5-turbo-1106 --examiner_model gpt-4-1106-preview 
```

3. Run the `examiner` task for the `diagnosis` section on all cases and using new student result from `gpt-4o-mini` model generated from the `student` task:
```
python main.py --task examiner --section diagnosis --case all --med_student_dataset ./output/med-student-with-gpt-4o-mini.json --med_exam_dataset ./data/med-exam.json --student_model gpt-4o-mini
```

### LangFuse
To use [LangFuse](https://www.langfuse.com/) in this project, you need to set the following environment variables:
```
LANGFUSE_SECRET_KEY=<YOUR_LANGFUSE_SECRET_KEY>
LANGFUSE_PUBLIC_KEY=<YOUR_LANGFUSE_PUBLIC_KEY>
LANGFUSE_HOST=<YOUR_LANGFUSE_HOST>
```

## Key Functions

- `load_data(dataset_path, is_examiner)`: Loads data from a JSON file. It takes a path to the dataset and a boolean indicating whether to load the examiner dataset.

- `save_result(path, dataset, is_examiner)`: Saves the updated dataset to a JSON file at the specified path.

- `parse_range(val)`: Parses a string that may represent a range (e.g., "1-10") or a single number (e.g., "5") and returns a tuple of integers.

- `run_model(model, prompt_template, input_data, pre_processing_func, post_processing_func, **kwargs)`: Executes the LLM with the given prompt template and input data. It includes optional pre- and post-processing functions.

- `llm_as_medical_student(*args, **kwargs)`: Simulates an LLM acting as a medical student on a dataset.

- `llm_as_examiner(*args, **kwargs)`: Simulates an LLM acting as an examiner on a dataset.

- `main(args)`: The main function that orchestrates the execution of the program based on the provided command-line arguments.

- `parse_args()`: Utilizes argparse to define and parse command-line arguments for the program.

## Notes
- The script uses the langchain library for interacting with the LLM. Ensure you have the necessary permissions and API keys if required by the library.
- Logging is implemented throughout the script. Use the `-v` or `--verbose` flag for more detailed logging information.


## Todo
- Fine-tune a Llama-examiner using GPT-4 examiner's instruction learning data
- Implement model name similar matching
- Add functionality for running models in batch