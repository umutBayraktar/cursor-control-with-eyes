from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import pyautogui
import cv2
import torch
from lib import detect_eyes
from utils import create_model, calculate_height_rate, calculate_width_rate
import numpy as np
import time

if __name__ == "__main__":
    device = torch.device(
	    'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = create_model(num_classes=2).to(device)
    model.load_state_dict(torch.load(
        f'../model17.pth', map_location=device
    ))
    model.eval()
    CLASSES = [
        'background', 'open'
    ]
    detection_threshold = 0.5
    screen_sizes = pyautogui.size()
    height = screen_sizes.height
    width = screen_sizes.width
    vid = cv2.VideoCapture(-1)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"height: {height}")
    y_rate = 0.13425925925925927#calculate_height_rate(model, width, height, vid)
    print(f"y_rate: {y_rate}")
    time.sleep(3)
    x_rate = 0.004166666666666667 #calculate_width_rate(model, width, height, vid)
    print(f"x_rate:{x_rate}")
    center_x = int(width/2)
    center_y = int(height/2)
    pyautogui.moveTo(center_x, center_y)
    print(f"cursor coordinates: x:{center_x} y:{center_y}")
    time.sleep(3)
    ret, first_frame = vid.read()
    cursor_x_prev_coord = center_x
    cursor_y_prev_coord = center_y
    if ret:
        eyes = detect_eyes(first_frame)
        for eye in eyes:
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
                draw_boxes = boxes.copy()
                eye_x = None
                eye_y = None
                for j, box in enumerate(draw_boxes):
                    if j == 0:
                        x_prev_coord = eye["y"] + box[0]
                        y_prev_coord = eye["x"] + box[1]

    #x_prev_coord = center_x
    #y_prev_coord = center_y
    print(f"x_prev_coord: {x_prev_coord}")
    print(f"y_prev_coord: {y_prev_coord}")
    print("Basla")
    while(True):
        ret, frame = vid.read()
        if ret:
            eyes = detect_eyes(frame)
            for eye in eyes:
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
                    draw_boxes = boxes.copy()
                    eye_x = None
                    eye_y = None
                    for j, box in enumerate(draw_boxes):
                        if j == 0:
                            eye_x = eye["y"] + box[0]
                            eye_y = eye["x"] + box[1]
                    if eye_x and eye_y:
                        #import pdb; pdb.set_trace()
                        diff_x = eye_x - x_prev_coord
                        diff_y = eye_y - y_prev_coord
                        distance_x = diff_x / x_rate
                        distance_y = diff_y / y_rate
                        if distance_x<0:
                            distance_x = distance_x * -1
                        if distance_y <0:
                            distance_y = distance_y * -1
                        distance_x = distance_x // width
                        distance_y = distance_y // height
                        if diff_x == 0 or diff_y == 0:
                            print("fark yok")
                            continue
                        if  diff_x > 0 : # saga git, arttir
                            print("sag diff_x>0")
                            if cursor_x_prev_coord != width or cursor_x_prev_coord != 0:
                                new_x_coord = cursor_x_prev_coord + int(distance_x)
                        elif diff_x < 0: #sola git azalt
                            print("sol diff_x<0")
                            if cursor_x_prev_coord != width or cursor_x_prev_coord != 0:
                                new_x_coord = cursor_x_prev_coord - int(distance_x)
                        if diff_y > 0: # asagi git, arttir
                            print("asagi diff_y>0")
                            if cursor_y_prev_coord != height or cursor_y_prev_coord != 0:
                                new_y_coord = cursor_y_prev_coord + int(distance_y)
                        elif diff_y  < 0: #yukari git, azalt
                            print("yukari diff_y<0")
                            if cursor_y_prev_coord != height or cursor_y_prev_coord != 0:
                                new_y_coord = cursor_y_prev_coord - int(distance_y)
                        print(f"eye_x: {eye_x}, eye_y: {eye_y}")
                        print(f"distance_x: {distance_x}, distance_y: {distance_y}")
                        print(f"x_prev_coord: {x_prev_coord} y_prev_coord : {y_prev_coord}")
                        print(f"new_x_coord: {new_x_coord}, new_y_coord: {new_y_coord}")
                        cursor_x_prev_coord = new_x_coord
                        cursor_y_prev_coord = new_y_coord
                        pyautogui.moveTo(new_x_coord, new_y_coord)

                        print(f"cursor coordinates: x:{cursor_x_prev_coord} y:{cursor_y_prev_coord}")
                        x_prev_coord = eye_x
                        y_prev_coord = eye_y
                        time.sleep(0.3)
