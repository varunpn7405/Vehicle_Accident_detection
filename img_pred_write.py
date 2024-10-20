import cv2,json,os
from ultralytics import YOLO

onnx_model = YOLO("best.onnx")
class_names=onnx_model.names

img_data_dict={}
path = os.getcwd()
inputPar = os.path.join(path, r'Dataset')
folders = os.listdir(inputPar)

for folder in folders:

    if folder in ["test"]:
        inputChild = os.path.join(inputPar, folder,"images")
        files = os.listdir(inputChild)

        for file in files:
            img_data_dict[file]=[]
            imgpath=os.path.join(inputChild,file)
            image = cv2.imread(imgpath)

            # Run inference
            results = onnx_model(image)

            # Extract predictions
            for result in results:
                boxes = result.boxes  # get bounding boxes
                
                for box in boxes:
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
                    conf = box.conf.item()  # Confidence score
                    class_id = int(box.cls.item())  # Class ID
                    clsName=class_names[class_id]
                    bbx_dict={"Bbox":[x1, y1, x2, y2],"class":f"{clsName}"}

                    if file in img_data_dict:
                        img_data_dict[file].append(bbx_dict)


with open("img_pred.json","w") as f:
    json.dump(img_data_dict,f,indent=4)




