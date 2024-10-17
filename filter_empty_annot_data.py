import os,shutil

# Function to check if a file is empty
def is_empty_file(file_path):
    return os.path.exists(file_path) and os.stat(file_path).st_size == 0

image_extensions = ['.jpg', '.jpeg', '.png']

path = os.getcwd()
inputPar = os.path.join(path, r'accident detection.v10i.yolov11')
outputPar = os.path.join(path, r'accident detection.v10i.yolov11_filtered')

if not os.path.exists(outputPar):
    os.makedirs(outputPar)

folders = os.listdir(inputPar)

for folder in folders:

    if folder in ["test","train","valid"]:
        inputChild = os.path.join(inputPar, folder,"labels")
        outputChild1 = os.path.join(outputPar, folder,"labels")

        if not os.path.exists(outputChild1):
            os.makedirs(outputChild1)

        outputChild2 = os.path.join(outputPar, folder,"images")

        if not os.path.exists(outputChild2):
            os.makedirs(outputChild2)

        files = os.listdir(inputChild)

        for file in files:
            annotation_path = os.path.join(inputChild, file)
            
            # Check if the annotation file is empty
            if not is_empty_file(annotation_path):
                shutil.copy(annotation_path,os.path.join(outputChild1,file))
                # Try to find and remove the corresponding image file
                image_name = os.path.splitext(file)[0]

                for ext in image_extensions:
                    image_path = os.path.join(inputPar,folder,"images", image_name + ext)

                    if os.path.exists(image_path):
                        shutil.copy(image_path,os.path.join(outputChild2,image_name + ext))
