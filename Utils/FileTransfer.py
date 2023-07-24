import os
import shutil

# CNN Model Face Recognition

save_path = "./Face/Raw_Images/NotFace"
os.makedirs(save_path, exist_ok=True)
parent_folder = "/Users/ryuaus26/Desktop/AcneApp/FaceRecon/natural_images"
folder_path = os.listdir(parent_folder)
folder_path.sort()

folder_list = []  # Create an empty list to store all subfolders

for folder in folder_path:
    folder_list.append(os.path.join(parent_folder, folder))

folder_list.sort()  # Sort the list of subfolders

for folder_name in folder_list:
    for file in os.listdir(folder_name):
        file_path = os.path.join(folder_name, file)
        if file_path.endswith('.jpg'):
            shutil.copy(file_path, save_path)
            print(f"Copied file {file} to {save_path}")
