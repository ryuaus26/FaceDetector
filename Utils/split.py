import os
import random
import shutil

# Define the source folders
face_folder = "./Face/Raw_Images/Face"
nonface_folder = "./Face/Raw_Images/NotFace"

# Define the destination folders
train_folder = "./Face/Raw_Images/Train"
val_folder = "./Face/Raw_Images/Val"

# Create the destination folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get the list of image files in the face and nonface folders
face_files = os.listdir(face_folder)
nonface_files = os.listdir(nonface_folder)

# Shuffle the image files
random.shuffle(face_files)
random.shuffle(nonface_files)

# Calculate the number of images for each split
face_train_count = int(len(face_files) * 0.95)
nonface_train_count = int(len(nonface_files) * 0.95)

# Move face images to train and val folders
for i in range(face_train_count):
    file = face_files[i]
    src = os.path.join(face_folder, file)
    dst = os.path.join(train_folder, "Face", file)
    shutil.copy(src, dst)

for i in range(face_train_count, len(face_files)):
    file = face_files[i]
    src = os.path.join(face_folder, file)
    dst = os.path.join(val_folder, "Face", file)
    shutil.copy(src, dst)

# Move nonface images to train and val folders
for i in range(nonface_train_count):
    file = nonface_files[i]
    src = os.path.join(nonface_folder, file)
    dst = os.path.join(train_folder, "NotFace", file)
    shutil.copy(src, dst)

for i in range(nonface_train_count, len(nonface_files)):
    file = nonface_files[i]
    src = os.path.join(nonface_folder, file)
    dst = os.path.join(val_folder, "NotFace", file)
    shutil.copy(src, dst)
