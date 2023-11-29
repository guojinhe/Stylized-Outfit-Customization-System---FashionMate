# Draw Landmarks func

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks

  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# Resize Image

import cv2

original_img = cv2.imread("Model\models\IMG_20231129_221906.jpg")

if original_img is None:
    print("Error: Unable to load the image.")

original_height, original_width, _ = original_img.shape
desired_width = 550
desired_height = int(desired_width*original_height/original_width)

resized_img = cv2.resize(original_img, (desired_width, desired_height))

cv2.imwrite("Model\models\IMG_20231129_221906.jpg", resized_img)
print("Resized image saved successfully.")

# Add Landmark

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='Model\models\pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file("Model\models\IMG_20231129_221906.jpg")
bgr_image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)

# STEP 4: Detect pose landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(bgr_image, detection_result)
cv2.imshow('Pose_Landmarks_Img',annotated_image)

cv2.waitKey(0) & 0xFF == ord('q')
cv2.destroyAllWindows()
# cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# Custumize 1 : calculate the length bewteen two shoulder (肩寬)

pose_landmarks_list = detection_result.pose_landmarks

for person_pose_landmarks in pose_landmarks_list:
        left_shoulder_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        print(f"Left Shoulder Coordinates: x={left_shoulder_landmark.x}, y={left_shoulder_landmark.y}, z={left_shoulder_landmark.z}")
        print(f"Right Shoulder Coordinates: x={right_shoulder_landmark.x}, y={right_shoulder_landmark.y}, z={right_shoulder_landmark.z}")

        shoulder_length = np.linalg.norm(np.array([left_shoulder_landmark.x-right_shoulder_landmark.x,left_shoulder_landmark.y-right_shoulder_landmark.y,left_shoulder_landmark.z-right_shoulder_landmark.z]))
        print(f"Shoulder_length: {shoulder_length}")