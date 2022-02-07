# Importing all required packages
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
def create_model(num_classes):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model


source_code_path = os.path.join(os.path.dirname(__file__))
base_path = os.path.join(source_code_path, "..")
cascades_path = os.path.join(base_path, "cascades")

images_path = os.path.join(source_code_path, "output_images")
video_file = "video.webm" #os.path.join(base_path, "2022-02-04-225432.webm")
eye_cascade_file = os.path.join(cascades_path, "haarcascade_eye.xml")


eye_cascade = cv2.CascadeClassifier(eye_cascade_file)



# create a function to detect eyes
def detect_eyes(img):
	
	eye_img = img.copy()
	eye_rect = eye_cascade.detectMultiScale(eye_img,
											scaleFactor = 1.2,
											minNeighbors = 5)
	eyes = []
	for (x, y, w, h) in eye_rect:
		cv2.rectangle(eye_img, (x, y),
					(x + w, y + h), (255, 255, 255), 3)
		print(x,y,w,h)
		eyes.append(eye_img[y:y+h, x:x+w])
	return eyes


if __name__ == "__main__":
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	model = create_model(num_classes=2).to(device)
	model.load_state_dict(torch.load(
        f'model17.pth', map_location=device
    ))
	model.eval()
	CLASSES = [
        'background', 'open'
    ]
	detection_threshold = 0.7
	vid = cv2.VideoCapture(video_file)
	counter = 1
	
	while(True):
		ret, frame = vid.read()
		if ret:
			copy_frame = frame.copy()
			eyes = detect_eyes(copy_frame)
			inner_count = 1
			for eye in eyes:
				output_file = os.path.join(images_path,f"frame_{counter}_eye_{inner_count}.png")
				predict_image = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB).astype(np.float32)
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
						cv2.rectangle(eye,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)

				cv2.imwrite(output_file ,eye)
				inner_count += 1
			counter+=1

