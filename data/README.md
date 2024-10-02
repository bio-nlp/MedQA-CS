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
    "opening": "Opening Scenario:\n\...",
    "chat_history": "N/A"
  },
  "ground_truth_output": "\"What con...",
  "prompt": {
    "_type": "prompt",
    "input_variables": [
      "opening",
      "chat_history"
    ],
    "template": "You are a doctor an..."
  },
  "output": {
    "gpt-3.5-turbo-1106": "When did ...",
    "gpt-4-1106-preview": "Can you d...",
    "claude-3-sonnet-20240229": "Can...",
    "claude-3-haiku-20240307": "Can ...",
    "claude-3-opus-20240229": "What ..."
  }
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

- `input`: An object containing the input data for evaluation, also include the generated output data from student task.

- `prompt`: An object containing the prompt:
    - `_type`: A string indicate this a prompt
    - `input_variables`: An array of input variable names used in the prompt template.
    - `template`: A prompt template that includes placeholders for the input variables.

- `result`: An object containing the model name for the input data, examiner model name, and the evaluation result.

### Example Case Object
```
{
  "unique_id": 1,
  "section": "qa",
  "case_id": 1,
  "conversation_turn_id": 1,
  "input": {
    "prev_conversation": "N/A",
    "opening": "Opening Scenario:\n\nJosep...",
    "question": {
      "gpt-4o": "Can you describe what the...",
      "gpt-4-1106-preview": "Can you descr...",
      "gpt-3.5-turbo-1106": "When did the ...",
      "claude-3-opus-20240229": "What brin...",
      "claude-3-sonnet-20240229": "Can you...",
      "claude-3-haiku-20240307": "Can you ..."
    },
    "ground_truth": "\"What concerns you m..."
  },
  "prompt": {
    "_type": "prompt",
    "input_variables": [
      "prev_conversation",
      "opening",
      "question",
      "ground_truth"
    ],
    "template": "As an evaluator for the U..."
  },
  "output": {
    "gpt-4o": {
      "gpt-4-1106-preview": "Score: 1\nRea...",
    },
    "gpt-4-1106-preview": {
      "gpt-4-1106-preview": "Score: 1\nRea...",
    },
    "gpt-3.5-turbo-1106": {
      "gpt-4-1106-preview": "Score: 1\nRea...",
    },
    "claude-3-opus-20240229": {
      "gpt-4-1106-preview": "Score: 0\nRea...",
    },
    "claude-3-sonnet-20240229": {
      "gpt-4-1106-preview": "Score: 1\nRea...",
    },
    "claude-3-haiku-20240307": {
      "gpt-4-1106-preview": "Score: 1\nRea..."
    }
  }
},
```
