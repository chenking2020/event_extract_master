import json

file_path = "/Users/mac/Desktop/dev_data.json"
with open(file_path, "r") as f:
    for line in f:
        line_dict = json.loads(line)
        subjects = []
        for spo in line_dict["spo_list"]:
            subject = spo["subject"]
            if subject not in subjects:
                subjects.append(subject)
        if len(subjects) > 1:
            print(line)
