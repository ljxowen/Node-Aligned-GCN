import os
import numpy as np

#function for load the data
def load_data(path):
    all_data = os.listdir(path)
    data = []
    IDC_label = []
    patients = []

    for patient in all_data:
        IDC_class = os.listdir(path + "/" + patient)
        for case in IDC_class:
            img_names = os.listdir(path + "/" + patient + "/" + case)
            for img_name in img_names:
                img_path = path + "/" + patient + "/" + case + "/" + img_name
                #store the data and label
                data.append(img_path)
                IDC_label.append(int(case))
                patients.append(patient)

    return data, IDC_label, patients



# function to mapping the data and patient
def create_patient_data_map(patients):
    patient_data_map = {}

    for index, patient_id in enumerate(patients):
        if patient_id not in patient_data_map:
            patient_data_map[patient_id] = [index]
        else:
            patient_data_map[patient_id].append(index)

    return patient_data_map



# generate the target label for each wsi, for classification
def create_target_label(patient_data_map, IDC_label):
    target_label_map = {}

    for patient, indices in patient_data_map.items():
        curr_labels = [IDC_label[i] for i in indices]
        labels_num = len(curr_labels)
        pos_num = curr_labels.count(1)
        ratio = pos_num / labels_num if labels_num > 0 else 0

        # give level based on positive case ratio
        if ratio >= 0.5: level = 1
        elif ratio < 0.5 : level = 0

        target_label_map[patient] = level

    return target_label_map



if __name__ == '__main__':
    pass