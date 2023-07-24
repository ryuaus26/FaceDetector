import os

folder_path = '/Users/ryuaus26/Desktop/AcneApp/FaceRecon/Face/Raw_Images/Images'

file_list = os.listdir(folder_path)



counter = 0

for filename in (file_list):
    file_extension = os.path.splitext(filename)[1]
    new_filename = f"Face-{counter}{file_extension}"
    os.rename(os.path.join(folder_path,filename),os.path.join(folder_path,new_filename))
    counter+=1

print(counter)