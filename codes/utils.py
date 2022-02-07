from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import pyautogui
import cv2
import torch
from lib import detect_eyes
import numpy as np

def create_model(num_classes):

    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def calculate_height_rate(model, width, height, capturer):
    half_width = int(width/2)
    detection_threshold = 0.5
    for i in range(0,height+20,20):
        pyautogui.moveTo(half_width,i) # move cursor TOP to BUTTOM
        ret, frame = capturer.read()
        print(ret)
        if ret:
            eyes = detect_eyes(frame)
            #print(eyes)
            for eye in eyes:
                print(f"eye info x :{eye.get('x')} y :{eye.get('y')}")
                predict_image = cv2.cvtColor(eye["img"], cv2.COLOR_BGR2RGB).astype(np.float32)
                predict_image /= 255.0
                predict_image = np.transpose(predict_image, (2, 0, 1)).astype(np.float)
                predict_image = torch.tensor(predict_image, dtype=torch.float).cuda()
                predict_image = torch.unsqueeze(predict_image, 0)
                with torch.no_grad():
                    outputs = model(predict_image)
                outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
                if len(outputs[0]['boxes']) != 0:
                    boxes = outputs[0]['boxes'].data.numpy()
                    scores = outputs[0]['scores'].data.numpy()
                    boxes = boxes[scores >= detection_threshold].astype(np.int32)
                    #print(len(boxes))
                    draw_boxes = boxes.copy()
                    eye_1_y = None
                    for j, box in enumerate(draw_boxes):
                        print(f"j: {j}")
                        if j == 0:    
                            eye_1 = box
                            eye_1_y = eye["x"] + eye_1[1]   
                    if eye_1_y:
                        if i == 0:
                            y_min = eye_1_y
                        y_max = eye_1_y

    print(f"y_min: {y_min}")
    print(f"y_max: {y_max}")
    y_rate = (y_max - y_min)/ height
    return y_rate

def calculate_width_rate(model, width, height, capturer):
    half_height = int(height/2)
    detection_threshold = 0.5
    for i in range(0,width+20,20):
        pyautogui.moveTo(i,half_height) # move cursor TOP to BUTTOM
        ret, frame = capturer.read()
        print(ret)
        if ret:
            eyes = detect_eyes(frame)
            #print(eyes)
            for eye in eyes:
                print(f"eye info x :{eye.get('x')} y :{eye.get('y')}")
                predict_image = cv2.cvtColor(eye["img"], cv2.COLOR_BGR2RGB).astype(np.float32)
                predict_image /= 255.0
                predict_image = np.transpose(predict_image, (2, 0, 1)).astype(np.float)
                predict_image = torch.tensor(predict_image, dtype=torch.float).cuda()
                predict_image = torch.unsqueeze(predict_image, 0)
                with torch.no_grad():
                    outputs = model(predict_image)
                outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
                if len(outputs[0]['boxes']) != 0:
                    boxes = outputs[0]['boxes'].data.numpy()
                    scores = outputs[0]['scores'].data.numpy()
                    boxes = boxes[scores >= detection_threshold].astype(np.int32)
                    #print(len(boxes))
                    draw_boxes = boxes.copy()
                    eye_1_x = None
                    for j, box in enumerate(draw_boxes):
                        print(f"j: {j}")
                        if j == 0:    
                            eye_1 = box
                            eye_1_x = eye["y"] + eye_1[0]   
                    if eye_1_x:
                        if i == 0:
                            x_min = eye_1_x
                        x_max = eye_1_x

    print(f"x_min: {x_min}")
    print(f"x_max: {x_max}")
    x_rate = (x_max - x_min)/ width
    return x_rate



