import os 
import cv2

source_code_path = os.path.join(os.path.dirname(__file__))
base_path = os.path.join(source_code_path, "..","..")
cascades_path = os.path.join(base_path, "cascades")

images_path = os.path.join(source_code_path, "output_images")
video_file = "video.webm"  # os.path.join(base_path, "2022-02-04-225432.webm")

eye_cascade_file = os.path.join(cascades_path, "haarcascade_eye.xml")
eye_cascade = cv2.CascadeClassifier(eye_cascade_file)


def detect_eyes(img):
	eye_img = img.copy()
	eye_rect = eye_cascade.detectMultiScale(eye_img,
											scaleFactor=1.2,
		            						minNeighbors=5)
	eyes = []
	for(x, y, w, h) in eye_rect[0:2]:
		tmp = {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "img": eye_img[y:y+h, x:x+w]}
		eyes.append(tmp)
	return eyes