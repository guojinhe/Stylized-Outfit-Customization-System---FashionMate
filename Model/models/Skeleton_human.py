
def Skeleton_Human() :

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

    # cv2.waitKey(0) & 0xFF == ord('q')
    # cv2.destroyAllWindows()
    # cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Calculate the joint proportions :

    joints_length = []

    pose_landmarks_list = detection_result.pose_landmarks 

    for person_pose_landmarks in pose_landmarks_list: # person_pose_list 是不同人的array,從其中抓每一個人出來算該人的關節資訊 (但通常只會有一個人)
            
        left_shoulder_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER] #11
        left_elbow_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW] #13
        left_wrist_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST] #15
        left_hip_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP] #23
        left_knee_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE] #25
        left_ankle_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE] #27

        right_shoulder_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER] #12
        right_elbow_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW] #14
        right_wrist_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST] #16
        right_hip_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP] #24
        right_knee_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE] #26
        right_ankle_landmark = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE] #28

        # print(f"Right Shoulder Coordinates: x={round(right_shoulder_landmark.x,2)}, y={round(right_shoulder_landmark.y,2)}, z={round(right_shoulder_landmark.z,2)}")

        def calculate_length(a,b):
            return np.linalg.norm(np.array([a.x-b.x,a.y-b.y]))

        joints_length.append(calculate_length(left_shoulder_landmark,right_shoulder_landmark)); # 0 Shoulder_length 11-12

        joints_length.append(calculate_length(right_wrist_landmark,right_elbow_landmark)); # 1 Right_SmallArm_length 14-16
        joints_length.append(calculate_length(right_elbow_landmark,right_shoulder_landmark)); # 2 Right_BigArm_length 12-14
        joints_length.append(calculate_length(right_shoulder_landmark,right_hip_landmark)); # 3 Right_Body_length 12-24

        joints_length.append(calculate_length(left_wrist_landmark,left_elbow_landmark)); # 4 Left_LittleArm_length 13-15
        joints_length.append(calculate_length(left_shoulder_landmark,left_elbow_landmark)); # 5 Left_BigArm_length 11-13
        joints_length.append(calculate_length(left_shoulder_landmark,left_hip_landmark)); # 6 Left_Body_length 11-23

        joints_length.append(calculate_length(right_hip_landmark,left_hip_landmark)); # 7 Hip_length 23-24

        joints_length.append(calculate_length(right_hip_landmark,right_knee_landmark)); # 8 Right_BigLeg_length 24-26
        joints_length.append(calculate_length(right_knee_landmark,right_ankle_landmark)); # 9 Right_SmallLeg_length 24-26

        joints_length.append(calculate_length(left_hip_landmark,left_knee_landmark)); # 10 Left_BigLeg_length 24-26
        joints_length.append(calculate_length(left_knee_landmark,left_ankle_landmark)); # 11 Left_SmallLeg_length 24-26


        print(f"0 Shoulder_length: {joints_length[0]}") 
        print(f"1 Right_SmallArm_length: {joints_length[1]}") 
        print(f"2 Right_BigArm_length: {joints_length[2]}") 
        print(f"3 Right_Body_length: {joints_length[3]}") 
        print(f"4 Left_LittleArm_length: {joints_length[4]}") 
        print(f"5 Left_BigArm_length: {joints_length[5]}") 
        print(f"6 Left_Body_length: {joints_length[6]}") 
        print(f"7 Hip_length: {joints_length[7]}") 
        print(f"8 Right_BigLeg_length: {joints_length[8]}") 
        print(f"9 Right_SmallLeg_length: {joints_length[9]}") 
        print(f"10 Left_BigLeg_length: {joints_length[10]}") 
        print(f"11 Left_SmallLeg_length: {joints_length[11]}") 
    
    return joints_length


# Skeleton_Human()