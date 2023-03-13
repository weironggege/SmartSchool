import cv2
from mediapipe.python.solutions import pose as mp_pose
import numpy as np

if __name__ == "__main__":

    video_cap = cv2.VideoCapture("./jump_outte.mp4")
    
    pose_tracker = mp_pose.Pose()
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fr = open("coor.txt", "a+")
    i = 0
    ori_x, ori_y = 0., 0.
    while True:

        sucess, inp_frame = video_cap.read()

        if not sucess:
            print("Error")
            break
        pro_frame = cv2.cvtColor(inp_frame, cv2.COLOR_BGR2RGB)

        result = pose_tracker.process(image=pro_frame)

        pose_landmarks = result.pose_landmarks

        if pose_landmarks is not None:

            pose_landmarks = np.array(
                                [[lmk.x * inp_frame.shape[1], lmk.y * inp_frame.shape[0], lmk.z * inp_frame.shape[1]]
                                for lmk in pose_landmarks.landmark])

            pose_landmarks = pose_landmarks[:, :2]

            right_h = pose_landmarks[30]

            if i == 0:
                ori_x, ori_y = right_h[0], right_h[1]
            
            res_coor = str((right_h[0] - ori_x) / 5.0) + " " + str((right_h[1] - ori_y) / 5.0)
            if right_h[1] - ori_y < 0:
                fr.write(res_coor + "\n")

        i += 1

        if cv2.waitKey(1) in [ord('q'), 27]:
            break
            
    
