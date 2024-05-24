import json
from collections import defaultdict
from run import Section


class DataViewer:
    def __init__(self, dataset_path):
        self.lookup = None
        self.dataset = None
        self.dataset_path = dataset_path
        self.open_dataset(dataset_path)

    def open_dataset(self, dataset_path):
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

        self.lookup = lookup
        self.dataset = data

        return data, lookup

    def get_data(self, section, case, turn=1):
        index = self.lookup[section][case][turn]
        data = self.dataset[index]
        return data

    def view_data(self, section, case, turn=1):
        data = self.get_data(section, case, turn)
        return json.dumps(data, indent=2)


if __name__ == "__main__":
    data_viewer = DataViewer("data/evaluation.json")
    print(data_viewer.view_data("diagnosis", 1))
