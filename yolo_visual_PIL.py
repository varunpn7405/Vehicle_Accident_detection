import os
from PIL import Image,ImageDraw,ImageFont

font = ImageFont.truetype("arial.ttf", 15)

path = os.getcwd()
inputPar = os.path.join(path, r'Dataset')
outputPar = os.path.join(path, r'Visualisation')

if not os.path.exists(outputPar):
    os.makedirs(outputPar)

folders = os.listdir(inputPar)

cls_clr = {"Accident":"#eb0523"}

clsfile = os.path.join(path, 'classes.txt')
with open(clsfile) as tf:
    clsnames = [cl.strip() for cl in tf.readlines()]

for folder in folders:

    if folder in ["test","train","valid"]:
        inputChild = os.path.join(inputPar, folder,"labels")
        outputChild = os.path.join(outputPar, folder)

        if not os.path.exists(outputChild):
            os.makedirs(outputChild)

        files = os.listdir(inputChild)
        for file in files:
            fileName, ext = os.path.splitext(file)
            finput = os.path.join(inputChild, file)

            with open(finput) as tf:
                Yolodata = tf.readlines()

            imgpath1 = os.path.join(inputPar,folder,"images", fileName + '.jpg')
            # imgpath2 = os.path.join(inputPar,folder,"images", fileName + '.png')
            
            if os.path.exists(imgpath1):
                imgpath=imgpath1

            # elif os.path.exists(imgpath2):
            #     imgpath=imgpath2

            if os.path.exists(imgpath):
                print("plotting >>",fileName + '.jpg')
                img = Image.open(imgpath)
                draw = ImageDraw.Draw(img)
                width, height = img.size

                for obj in Yolodata:
                    clsName = clsnames[int(obj.split(' ')[0])]
                    xnew = float(obj.split(' ')[1])
                    ynew = float(obj.split(' ')[2])
                    wnew = float(obj.split(' ')[3])
                    hnew = float(obj.split(' ')[4])
                    label=f"{clsName}"

                    # box size
                    dw = 1 / width
                    dh = 1 / height

                    # coordinates
                    xmax = int(((2 * xnew) + wnew) / (2 * dw))
                    xmin = int(((2 * xnew) - wnew) / (2 * dw))
                    ymax = int(((2 * ynew) + hnew) / (2 * dh))
                    ymin = int(((2 * ynew) - hnew) / (2 * dh))

                    clr = cls_clr[clsName]
                    tw, th = font.getbbox(label)[2:]

                    # draw bbox and classname::
                    draw.rectangle([(xmin,ymin),(xmax,ymax)],outline =clr,width=2)
                    txtbox = [(xmin,ymin-th),(xmin+tw,ymin)]
                    draw.rectangle(txtbox, fill=clr)
                    draw.text((xmin,ymin-th),label,fill='white',font=font)

                fout = os.path.join(outputChild, imgpath.split("\\")[-1])
                img.save(fout)

            else:
                print(f'{imgpath}  >> img not found:')






