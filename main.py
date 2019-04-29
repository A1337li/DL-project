from csv_reader import get_labels

labels = get_labels("Data_Osteo_Tiles/ML_Features_1144.csv")
label_counter = [0]*3

for label in labels.values():
	label_counter[label] += 1

print("label percentages: ")
for counter in label_counter:
	print(counter/len(labels))

