# Importing all required packages
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pyautogui


def create_model(num_classes):

    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


source_code_path = os.path.join(os.path.dirname(__file__))
base_path = os.path.join(source_code_path, "..")
cascades_path = os.path.join(base_path, "cascades")

images_path = os.path.join(source_code_path, "output_images")
video_file = "video.webm"  # os.path.join(base_path, "2022-02-04-225432.webm")
eye_cascade_file = os.path.join(cascades_path, "haarcascade_eye.xml")


eye_cascade = cv2.CascadeClassifier(eye_cascade_file)


# create a function to detect eyes
def detect_eyes(img):

	eye_img = img.copy()
	eye_rect = eye_cascade.detectMultiScale(eye_img,
											scaleFactor=1.2,
											minNeighbors=5)
	eyes = []
	for(x, y, w, h) in eye_rect:
		cv2.rectangle(eye_img, (x, y), (x + w, y + h), (255, 255, 255), 3)
		tmp = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "img": eye_img[y:y+h, x:x+w]}
		eyes.append(tmp)
	return eyes

def prepare_height_settings(model, height, width, capturer):
	screen_sizes = pyautogui.size()
	counter = 1
	half_width = 900
	x_y_coordinates = []
	cursor_counter = 0
	x_min = 0
	x_max = 0
	y_min = 0
	y_max = 0
	while(True):
		ret, frame = capturer.read()
		if ret:
			copy_frame = frame.copy()
			pyautogui.moveTo(half_width, cursor_counter)
			eyes = detect_eyes(copy_frame)
			inner_count = 1
			for eye in eyes:
				output_file = os.path.join(images_path,f"frame_{counter}_eye_{inner_count}.png")
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
					pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
					pred_label ="close"
					pred_x = 0
					pred_y = 0
					for j, box in enumerate(draw_boxes):
						x_y_coordinates.append((eye["y"]+box[0], eye["x"]+box[1]))
						print(j, (eye["y"]+box[0], eye["x"]+box[1]))
						if cursor_counter == 0:
							x_min = eye["y"]+box[0]
							y_min = eye["x"]+box[1]
						if (cursor_counter +20) > height:
							x_max = eye["y"]+box[0]
							y_max = eye["x"]+box[1]
						cv2.rectangle(eye["img"],
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)

				cv2.imwrite(output_file ,eye["img"])
				inner_count += 1
			counter+=1
			cursor_counter+=20
			if cursor_counter>height:
				break
	x_diff = x_max - x_min
	y_diff = y_max - y_min
	x_range = x_diff / width
	y_range = y_diff / height
	return x_range,y_range


if __name__ == "__main__":
	device = torch.device(
	    'cuda') if torch.cuda.is_available() else torch.device('cpu')
	model = create_model(num_classes=2).to(device)
	model.load_state_dict(torch.load(
        f'model17.pth', map_location=device
    ))
	model.eval()
	CLASSES = [
        'background', 'open'
    ]
	detection_threshold = 0.7
	screen_sizes = pyautogui.size()
	height = screen_sizes.height
	width = screen_sizes.width
	vid = cv2.VideoCapture(-1)
	counter = 1
	pyautogui.moveTo(int(width/2), int(height/2))
	#pyautogui.moveTo(half_width, height, 10)
	x_range, y_range = prepare_height_settings(model, height, width, vid)
	x_y_coordinates = []
	cursor_counter = 0
	eye_prev_x = 0
	eye_prev_y = 0
	while(True):
		#print("dongude")
		ret, frame = vid.read()
		if ret:
			copy_frame = frame.copy()
			eyes = detect_eyes(copy_frame)
			#print(eyes)
			inner_count = 1
			for eye in eyes:
				output_file = os.path.join(images_path,f"frame_{counter}_eye_{inner_count}.png")
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
					pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
					pred_label ="close"
					pred_x = 0
					pred_y = 0
					for j, box in enumerate(draw_boxes):
						x_y_coordinates.append((eye["y"]+box[0], eye["x"]+box[1]))
						print(j, (eye["y"]+box[0], eye["x"]+box[1]))
						eye_x = eye["y"]+box[0]
						eye_y = eye["x"]+box[1]
						eye_diff_x = eye_prev_x - eye_x
						eye_diff_y = eye_prev_y - eye_y
						eye_prev_x = eye["y"]+box[0]
						eye_prev_y = eye["x"]+box[1]
						# cv2.rectangle(eye["img"],
                        #     (int(box[0]), int(box[1])),
                        #     (int(box[2]), int(box[3])),
                        #     (0, 0, 255), 2)

				# cv2.imwrite(output_file ,eye["img"])
				inner_count += 1
			counter+=1
			cursor_counter+=20
			if cursor_counter>height:
				break