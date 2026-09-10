import os
import shutil

# CHANGE THESE PATHS
protocol_file = r"LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

audio_folder = r"LA/ASVspoof2019_LA_train/flac"

output_real = r"dataset/real"
output_fake = r"dataset/fake"

# create folders
os.makedirs(output_real, exist_ok=True)
os.makedirs(output_fake, exist_ok=True)

# limit (optional)
max_real = 500
max_fake = 500

count_real = 0
count_fake = 0


with open(protocol_file, "r") as f:

    for line in f:

        parts = line.strip().split()

        file_name = parts[1] + ".flac"

        label = parts[-1]

        source = os.path.join(audio_folder, file_name)


        if label == "bonafide" and count_real < max_real:

            shutil.copy(source, output_real)

            count_real += 1


        elif label == "spoof" and count_fake < max_fake:

            shutil.copy(source, output_fake)

            count_fake += 1


        if count_real >= max_real and count_fake >= max_fake:

            break


print("DONE")
print("Real:", count_real)
print("Fake:", count_fake)