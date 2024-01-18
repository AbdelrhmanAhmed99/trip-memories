import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import cv2
import json
import numpy as np
import sys
sys.path.append('/content/trip-memories/SPIGA')
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
import copy
from spiga.demo.visualize.plotter import Plotter
import matplotlib.pyplot as plt
import os
import json
import math
import copy
import numpy as np
import cv2
import torch
import torch.nn as nn
from skimage import io
import skimage
sys.path.append('/content/trip-memories')
from emonet.emonet.models import EmoNet
from torchvision import transforms
from emonet.emonet.data_augmentation import DataAugmentor
import pickle
class FaceProcessor:
    def __init__(self, det_thresh=0.7):
        self.det_thresh = det_thresh
        self.n_expression=5
        image_size = 256
        self.transform_image = transforms.Compose([transforms.ToTensor()])
        self.transform_image_shape_no_flip = DataAugmentor(image_size, image_size)
        self.load_models()
    def load_models(self):
        self.device = 'cuda:0'
        state_dict_path =f'/content/trip-memories/emonet/pretrained/emonet_{self.n_expression}.pth'
        print(f'Loading the emonet model from {state_dict_path}.')
        state_dict = torch.load(str(state_dict_path), map_location='cpu')
        state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
        self.net = EmoNet(n_expression=self.n_expression).to(self.device)
        self.net.load_state_dict(state_dict, strict=False)
        self.app = FaceAnalysis(allowed_modules=['detection'], det_thresh=self.det_thresh)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        dataset = '300wpublic'
        destination_dir = '/content/trip-memories/SPIGA/spiga/models/weights'
        # Destination file name
        destination_file = f'spiga_{dataset}.pt'

        # Create the destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)

        # Download the file using gdown
        file_id = '1YrbScfMzrAAWMJQYgxdLZ9l57nmTdpQC'
        file_url = f'https://drive.google.com/uc?id={file_id}'
        file_path = os.path.join(destination_dir, destination_file)

        os.system(f'gdown --id {file_id} -O "{file_path}"')

        self.processor = SPIGAFramework(ModelConfig(dataset))
        print(f"File downloaded and saved to {destination_dir}")
    def _rotate(self, point, angle, origin):
        x, y = point
        ox, oy = origin

        # Invert the y-coordinates
        y = -y
        oy = -oy

        qx = ox + math.cos(angle) * (x - ox) - math.sin(angle) * (y - oy)
        qy = oy + math.sin(angle) * (x - ox) + math.cos(angle) * (y - oy)

        # Invert the y-coordinates back
        qy = -qy

        return qx, qy

    def _process_faces(self,image, converted_bboxes, features):
        result_dict = {}

        for i, converted_bbox in enumerate(converted_bboxes):
            # Prepare variables
            x0, y0, w, h = converted_bbox
            face_image = copy.deepcopy(image[y0:y0+h, x0:x0+w])
            landmarks = np.array(features['landmarks'][i])
            headpose = np.array(features['headpose'][i])

            # Rotate face based on head pose angle
            rotation_angle = headpose[2]
            (h, w) = face_image.shape[:2]
            center = (w // 2, h // 2)
            new_landmarks = []
            for landmark in landmarks:
                # Adjust landmarks based on the position of the face in the original image
                adjusted_x = landmark[0] - x0
                adjusted_y = landmark[1] - y0
                # Rotate the adjusted landmarks
                new_x, new_y = self._rotate((adjusted_x, adjusted_y), math.radians(rotation_angle), center)
                new_landmarks.append((new_x, new_y))
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            rotated_face = cv2.warpAffine(face_image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
            headpose[2] = 0
            # Save rotated face image directly in the dictionary
            result_dict[f"face_{i}"] = {
                'image': rotated_face,
                'landmarks': new_landmarks,
                'headpose': headpose.tolist(),
                'bounding_box': converted_bbox
            }
        return result_dict
    def capture_frames(self,video_path, fps):
        video = cv2.VideoCapture(video_path)
        video_fps = video.get(cv2.CAP_PROP_FPS)
        if fps<1 : frame_skip=1
        else: frame_skip = int(video_fps / fps)

        frame_count = 0
        results = {}

        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                second = frame_count // video_fps
                frame = np.array(frame)
                result = self.emotion_analysis(frame)
                if result == None :
                    frame_count += 1
                    continue
                results[f'frame_{frame_count}'] = result
            frame_count += 1

        video.release()
        return results
    def process_faces(self,image):
        # Detection and preparation steps
        faces = self.app.get(image)
        box = [face.bbox.astype(int) for face in faces]
        # Conversion step
        converted_bboxes = self._convert_bboxes(image,box)
        if converted_bboxes == []:
            return None
        # Feature extraction
        features = self.processor.inference(image, converted_bboxes)

        # Face processing step
        result = self._process_faces(image,converted_bboxes, features)

        return result

    def _convert_bboxes(self,image, original_bboxes):
        converted_bboxes = []

        for original_bbox in original_bboxes:
            left, top, right, bottom = original_bbox
            width = right - left
            height = bottom - top
            margin = 20

            # Check if the converted bounding box exceeds image dimensions
            converted_left = max(left - margin, 0)
            converted_top = max(top - margin, 0)
            converted_right = min(right + margin, image.shape[1])
            converted_bottom = min(bottom + margin, image.shape[0])

            converted_width = converted_right - converted_left
            converted_height = converted_bottom - converted_top

            converted_bbox = [converted_left, converted_top, converted_width, converted_height]
            converted_bboxes.append(converted_bbox)

        return converted_bboxes

    def emotion_analysis(self,image):
      result= self.process_faces(image)
      if result==None:
        return None
      for k,v in result.items():
            image_np=v['image']
            image_height, image_width, _ = image_np.shape
            bounding_box = [0, 0, image_width, image_height]
            image1, _ = self.transform_image_shape_no_flip(image_np, bb=bounding_box)
            image2 = np.ascontiguousarray(image1)
            image3 = self.transform_image(image2)
            image=image3.to(self.device)
            batch_image = image.unsqueeze(0)
            self.net.eval()
            with torch.no_grad():
              out = self.net(batch_image)
            exp=np.argmax(out['expression'].cpu()[0])
            val=out['valence'].cpu()[0]
            aro=out['arousal'].cpu()[0]
            emotions=['Neutral','Happy','Sad','Surprise','Fear,Disgust,Anger']
            expl= emotions[exp]
            result[k]['expression']=expl
            result[k]['valence']=val
            result[k]['arousal']=aro
      return result
    def video_summary(self, video_path, fps, n):
        # Capture frames and perform emotion analysis
        results = self.capture_frames(video_path, fps)

        # Check if there are no frames processed
        if results is None or len(results) == 0:
            return None

        # Calculate average intensity for each frame
        frame_intensities = []
        for frame_key, frame_data in results.items():
            # Calculate average intensity for all faces in the frame
            face_intensities = []
            for face_key, face_data in frame_data.items():
                valence = face_data['valence']
                arousal = face_data['arousal']
                intensity = np.sqrt(valence**2 + arousal**2)
                face_intensities.append(intensity)

            # Calculate average intensity for the frame
            avg_intensity = np.mean(face_intensities)
            frame_intensities.append((frame_key, avg_intensity))

        # Sort frames based on average intensity in descending order
        sorted_frames = sorted(frame_intensities, key=lambda x: x[1], reverse=True)

        # Get top n frames or all frames if n is greater than the total frames

        selected_frames = sorted_frames[:n] if (n <= len(sorted_frames) and n>0) else sorted_frames

        # Retrieve the frames based on their keys

        frame_keys=[fk for fk,_ in selected_frames]

        summary_frames = {key: results[key] for key, _ in selected_frames}

        video = cv2.VideoCapture(video_path)
        video_fps = video.get(cv2.CAP_PROP_FPS)
        if fps<1 : frame_skip=1
        else: frame_skip = int(video_fps / fps)

        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                frame = np.array(frame)
                if f'frame_{frame_count}' in frame_keys:
                      results[f'frame_{frame_count}']['frame_image']=frame
            frame_count += 1
        video.release()
        with open(f'{os.path.splitext(video_path)[0]}.pkl', 'wb') as f:
          pickle.dump(summary_frames, f, protocol=pickle.HIGHEST_PROTOCOL)

        return summary_frames
