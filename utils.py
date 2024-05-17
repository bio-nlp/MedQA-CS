import json
import re


def extract_json_data(text):
    json_text = re.search(r"```json\n?({[\w\W]+})[\n]?```", text)  # find ```json{}```
    if json_text:
        return json.loads(json_text.group(1))
    else:
        json_text = re.search(r"{[\w\W]+}", text)  # only find {}
        if json_text:
            return json.loads(json_text.group(0))
        else:
            print("No JSON found in text")
            return None


def readable_json(text):
    return json.dumps(extract_json_data(text), indent=2)


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


def medical_student_physical_exam_post_processing(text):
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


def medical_student_diagnosis_post_processing(text):
    print(type(text))
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


def examiner_diagnosis_post_processing(text):
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
        max_score = 40 + 9 + finding_count
        result_dict["total score"] = f"{total_score}/{max_score}"
        result_dict["accuracy"] = "{:.2%}".format(total_score / max_score)

    text = extract_json_data(text)
    return json.dumps(text, indent=2)
