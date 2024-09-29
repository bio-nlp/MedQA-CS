import json
import re
import json_repair


def extract_json_data(text):
    json_text = re.search(r"```json\n?({[\w\W]+})[\n]?```", text)  # find ```json{}```
    if json_text:
        return json_repair.loads(json_text.group(1))
    else:
        json_text = re.search(r"{[\w\W]+}", text)  # only find {}
        if json_text:
            return json_repair.loads(json_text.group(0))
        else:
            print("No JSON found in text")
            return None


def readable_json(data):
    text = data["output"]
    return json.dumps(extract_json_data(text), indent=2)


def output_only_post_processing(data: dict) -> str:
    """
    A post-processing function that returns only the output from the data dictionary.

    Args:
        data (dict): A dictionary containing 'prompt', 'input', and 'output' keys.

    Returns:
        str: The value associated with the 'output' key.
    """
    return data["output"]


# def medical_student_closure_post_processing(text):
#     closure_json = extract_json_data(text)
#     output = (
#         "Closure: \n"
#         + closure_json["closure"]
#         + "\n\nQuestion:"
#         + challenge_question
#         + "\nAnswer: "
#         + closure_json["question"]
#     )
#     return output


def medical_student_qa_post_processing(data):
    text = data["output"]
    qa_json = extract_json_data(text)

    if not qa_json:
        return "No JSON found in text"

    return qa_json["question"]


def medical_student_physical_exam_post_processing(data):
    text = data["output"]
    physical_exams_json = extract_json_data(text)
    output = ""
    for key in physical_exams_json.keys():
        output += (
            physical_exams_json[key]["physical exam"]
            + ": "
            + physical_exams_json[key]["maneuver"]
            + "\n"
        )
        output += "reason: " + physical_exams_json[key]["reason"] + "\n"

    return output


def medical_student_diagnosis_post_processing(data):
    text = data["output"]
    diagnosis_json = extract_json_data(text)
    output = ""
    index = 1
    for key in diagnosis_json.keys():
        output += (
            "Diagnosis #" + str(index) + ": " + diagnosis_json[key]["diagnosis"] + "\n"
        )
        output += "Historical Finding(s): \n"
        if type(diagnosis_json[key]["Historical Findings"]) == str:
            output += "N/A \n"
        else:
            for hist_finding in diagnosis_json[key]["Historical Findings"]:
                output += hist_finding + "\n"
        output += "\n"

        output += "Historical reasons: \n"
        if type(diagnosis_json[key]["Historical reasons"]) == str:
            output += "N/A \n"
        else:
            for hist_finding in diagnosis_json[key]["Historical reasons"]:
                output += hist_finding + "\n"
        output += "\n"

        output += "Physical Exam Finding(s): \n"
        if type(diagnosis_json[key]["Physical exam data"]) == str:
            output += "N/A \n"
        else:
            for hist_finding in diagnosis_json[key]["Physical exam data"]:
                output += hist_finding + "\n"
        output += "\n"

        output += "Physical exam data reasons: \n"
        if type(diagnosis_json[key]["Physical exam data reasons"]) == str:
            output += "N/A \n"
        else:
            for hist_finding in diagnosis_json[key]["Physical exam data reasons"]:
                output += hist_finding + "\n"
        output += "\n\n"
        index += 1

    return output


def examiner_diagnosis_post_processing(data):
    def get_num_physical_finding(text):
        lines = text.strip().split("\n")
        count = 0
        in_physical_part = False
        for line in lines:
            if "Physical Exam Finding" in line:
                # print("start:", line)
                in_physical_part = True
                continue
            if in_physical_part:
                if not line:
                    # print("end:", line)
                    in_physical_part = False
                elif "N/A" not in line or "None" not in line:
                    # print(line)
                    count += 1

        # print(count)
        count = min(count, 9)
        return count

    def replace_total_score(result_dict, finding_count):
        total_score = result_dict["total score"]
        max_score = 30 + 9 + finding_count + 10
        result_dict["total score"] = f"{total_score}/{max_score}"
        result_dict["accuracy"] = "{:.2%}".format(total_score / max_score)

    result_dict = extract_json_data(data["output"])
    num_physical_finding = get_num_physical_finding(data["input"]["target"])
    replace_total_score(result_dict, num_physical_finding)

    return json.dumps(result_dict, indent=2)
