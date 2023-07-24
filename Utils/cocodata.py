import requests
import os

# instantiate COCO specifying the annotations json path
json_path = 'face_detection.json'
img_path = './Face/Images'

os.makedirs(img_path, exist_ok=True)

with open(json_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        line = line.replace('null', 'None') 
        data = eval(line)  # Convert the string representation of a dictionary to a dictionary object
        content = data['content']
        annotations = data['annotation']
        
        # Download the image
        image_name = content.split('/')[-1]
        image_path = os.path.join(img_path, image_name)
        response = requests.get(content)
        with open(image_path, 'wb') as image_file:
            image_file.write(response.content)
        
        