import os,shutil

image_extensions = ['.jpg', '.jpeg', '.png']

path = os.getcwd()
inputPar = os.path.join(path, r'accident detection.v10i.yolov11')
outputPar = os.path.join(path, r'accident detection.v10i.yolov11(Filtered))')

if not os.path.exists(outputPar):
    os.makedirs(outputPar)

folders = os.listdir(inputPar)

clsfile = os.path.join(path, 'classes copy.txt')

with open(clsfile) as tf:
    clsnames = [cl.strip() for cl in tf.readlines()]

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
            fileName, ext = os.path.splitext(file)
            finput = os.path.join(inputChild, file)
            
            with open(finput) as tf:
                Yolodata = tf.readlines()
            
            # for obj in Yolodata:
            if not all(int(obj.split(' ')[0])==2 or int(obj.split(' ')[0])==1 for obj in Yolodata):
                print(file)

                new_yolo_data=[]

                for obj in Yolodata:

                    if not (int(obj.split(' ')[0])==2 or int(obj.split(' ')[0])==1) :
                        new_yolo_data.append(obj)

                with open(os.path.join(outputChild1,file),"w") as tf:
                    tf.writelines(new_yolo_data)      

                image_name = os.path.splitext(file)[0]

                for ext in image_extensions:
                    image_path = os.path.join(inputPar,folder,"images", image_name + ext)

                    if os.path.exists(image_path):
                        shutil.copy(image_path,os.path.join(outputChild2,image_name + ext))
                        break
