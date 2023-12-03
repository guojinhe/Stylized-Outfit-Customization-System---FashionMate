# Pictures File Path = "Model\models\Skeleton_human\Input_Pictures"

def skeleton_human(Img_Path) :

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

    original_img = cv2.imread(Img_Path)

    if original_img is None:
        print("Error: Unable to load the image.")

    original_height, original_width, _ = original_img.shape
    desired_width = 500
    desired_height = int(desired_width*original_height/original_width)

    resized_img = cv2.resize(original_img, (desired_width, desired_height))

    cv2.imwrite(Img_Path, resized_img)
    # print("Resized image saved successfully.")

    # Add Landmark

    # STEP 1: Import the necessary modules.
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # STEP 2: Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path='Model\models\Skeleton_human\pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(Img_Path)
    bgr_image = cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR)

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(bgr_image, detection_result)
    cv2.imshow(Img_Path.split("\\")[-1]+"_Landmarked",annotated_image)

    # cv2.waitKey(0) & 0xFF == ord('q')
    # cv2.destroyAllWindows()
    # cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Calculate the joint proportions :

    pose_landmarks_list = detection_result.pose_landmarks 

    joints_coordinates = {}
    j2j_length = {}

    for person_pose_landmarks in pose_landmarks_list: # person_pose_list 是不同人的array,從其中抓每一個人出來算該人的關節資訊 (但通常只會有一個人，超過兩人會報錯)

        joints = ["LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST","LEFT_HIP","LEFT_KNEE","LEFT_ANKLE","RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST","RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"]
    
        joints_coordinates["LEFT_SHOULDER"] = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
        joints_coordinates["LEFT_ELBOW"] = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        joints_coordinates["LEFT_WRIST"] = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        joints_coordinates["LEFT_HIP"] = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
        joints_coordinates["LEFT_KNEE"] = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
        joints_coordinates["LEFT_ANKLE"] = person_pose_landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]

        joints_coordinates["RIGHT_SHOULDER"] = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        joints_coordinates["RIGHT_ELBOW"] = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
        joints_coordinates["RIGHT_WRIST"] = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        joints_coordinates["RIGHT_HIP"] = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP]
        joints_coordinates["RIGHT_KNEE"] = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_KNEE]
        joints_coordinates["RIGHT_ANKLE"] = person_pose_landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]

        #for i in range(len(joints)):
            #print(f"{joints[i]} Coordinates: ({round(joints_coordinates[joints[i]].x,2)}, {round(joints_coordinates[joints[i]].y,2)}, {round(joints_coordinates[joints[i]].z,2)})")

        def calculate_length(a,b):
            return np.linalg.norm(np.array([a.x-b.x,a.y-b.y]))
        
        for j1 in joints:
            for j2 in joints:
                j2j_length[(j1,j2)] = calculate_length(joints_coordinates[j1],joints_coordinates[j2]);
                #if(j2j_length[(j1,j2)]!=0) : 
                    #print(f"length[{j1},{j2}] = {round(j2j_length[(j1,j2)],3)}")

    return joints_coordinates, j2j_length

            
