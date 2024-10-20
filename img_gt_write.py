import os,json
from PIL import Image

img_data_dict={}
path = os.getcwd()
inputPar = os.path.join(path, r'Dataset')
folders = os.listdir(inputPar)

clsfile = os.path.join(path, 'classes.txt')

with open(clsfile) as tf:
    clsnames = [cl.strip() for cl in tf.readlines()]

for folder in folders:

    if folder in ["test"]:
        inputChild = os.path.join(inputPar, folder,"images")
        files = os.listdir(inputChild)

        for file in files:
            imgpath=os.path.join(inputChild,file)
            img_data_dict[file]=[]
            fileName, ext = os.path.splitext(file)
            finput = os.path.join(inputPar,folder,"labels", fileName + '.txt')

            with open(finput) as tf:
                Yolodata = tf.readlines()
                
            if os.path.exists(imgpath):
                print("plotting >>",fileName + '.jpg')
                img = Image.open(imgpath)
                width, height = img.size

                for obj in Yolodata:
                    clsName = clsnames[int(obj.split(' ')[0])]
                    xnew = float(obj.split(' ')[1])
                    ynew = float(obj.split(' ')[2])
                    wnew = float(obj.split(' ')[3])
                    hnew = float(obj.split(' ')[4])
                    # box size
                    dw = 1 / width
                    dh = 1 / height
                    # coordinates
                    xmax = int(((2 * xnew) + wnew) / (2 * dw))
                    xmin = int(((2 * xnew) - wnew) / (2 * dw))
                    ymax = int(((2 * ynew) + hnew) / (2 * dh))
                    ymin = int(((2 * ynew) - hnew) / (2 * dh))

                    bbx_dict={"Bbox":[xmin,ymin,xmax,ymax],"class":f"{clsName}"}

                    if file in img_data_dict:
                        img_data_dict[file].append(bbx_dict)

            else:
                print(f'{imgpath}  >> img not found:')

with open("img_gt.json","w") as f:
    json.dump(img_data_dict,f,indent=4)




