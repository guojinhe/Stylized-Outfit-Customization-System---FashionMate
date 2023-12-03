from Skeleton_Human import skeleton_human
import os
import cv2

source_path = "Model\models\Skeleton_human\Input_Pictures"

for filename in os.listdir(source_path) :
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        img_path = os.path.join(source_path, filename)
        print(f"File name: {filename}")
        joints_coordinates, j2j_length = skeleton_human(img_path)
        joints = ["LEFT_SHOULDER","LEFT_ELBOW","LEFT_WRIST","LEFT_HIP","LEFT_KNEE","LEFT_ANKLE","RIGHT_SHOULDER","RIGHT_ELBOW","RIGHT_WRIST","RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE"]
        is_full_body = True # 是否全身入鏡
        for i in joints:
            print(f"{i} coordinate: ({round(joints_coordinates[i].x,2)}, {round(joints_coordinates[i].y,2)})")
            if (joints_coordinates[i].x<0.0) or (joints_coordinates[i].x>1.0) or (joints_coordinates[i].y<0.0) or (joints_coordinates[i].y>1.0) :
                is_full_body = False;
        for i in joints:
            for j in joints:
                if not (i == j):
                    print(f"length({i},{j}) = { round(j2j_length[(i,j)],2) }") # j2j_length 是 dict, key是tuple型態=(第一關節, 第二關節)
        print('')
        if not is_full_body : 
            print("Reminder: This picture doesn't cover your whole body. For optimization, please upload your full body picture again!\n")

        cv2.waitKey(0)
