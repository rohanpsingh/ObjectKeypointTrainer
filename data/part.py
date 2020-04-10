import glob, os
import shutil
import re
import sys

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = sys.argv[2]
print("dataset directory: ", dataset_dir)

img_data = 'frames'

percentage_test = int(sys.argv[1])
print("percentange of valid images: ", percentage_test)

file_train = open(os.path.join(dataset_dir, 'train.txt'), 'w')
file_valid = open(os.path.join(dataset_dir, 'valid.txt'), 'w')

counter = 1  
index_test = round(100 / percentage_test)

for pathAndFilename in glob.iglob(os.path.join(dataset_dir, img_data, "*.jpg")):  
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    num = re.split("_",title)
    if counter == index_test:
        counter = 1
        file_valid.write(os.path.join(current_dir, dataset_dir, img_data, title+ext) + "\n")
    else:
        counter = counter + 1
        file_train.write(os.path.join(current_dir, dataset_dir, img_data, title+ext) + "\n")
