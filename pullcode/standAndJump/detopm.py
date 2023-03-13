from mediapipe.python.solutions import pose as mp_pose
import cv2
import numpy as np
import random


landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

require_landids = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 33, 34]
require_lines = [[0,33], [33,11], [33,12], [11,13], [12,14], [13,15], [14,16], [33,34], [34,23], [34,24], [23,25], [25,27], [24,26], [26,28]]


def get_land_line_dict():
    land_c, line_c = {}, {}
    for ridx in require_landids:
        land_c[ridx] = random.sample(list(np.arange(256)), 3)

    for liid in range(len(require_lines)):
        line_c[liid] = random.sample(list(np.arange(256)), 3)
    return land_c, line_c



if __name__ == "__main__":
    

    video_cap = cv2.VideoCapture("../ropeSkip/detectrope/ropev2.mp4")


    nameW = 'pose show'
    cv2.namedWindow(nameW, cv2.WINDOW_NORMAL)


    pose_tracker = mp_pose.Pose()
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = './rope_out.mp4' 

    out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps,
                           (video_width, video_height))





    point_c, line_c = get_land_line_dict()

    pose_tracker = mp_pose.Pose()

    while True:

        sucess, inp_frame = video_cap.read()
        if not sucess:
            print("Error")
            break

        pro_frame = cv2.cvtColor(inp_frame, cv2.COLOR_BGR2RGB)

        result = pose_tracker.process(image=pro_frame)

        pose_landmarks = result.pose_landmarks

        idx_points = {}

        if pose_landmarks is not None:

            pose_landmarks = np.array(
                                    [[lmk.x * video_width, lmk.y * video_height, lmk.z * video_width]
                                    for lmk in pose_landmarks.landmark])

            pose_landmarks = pose_landmarks[:, :2]

        
            for idx in require_landids:
                if idx == 33:
                    land = (pose_landmarks[11] + pose_landmarks[12]) * 0.5
                elif idx == 34:
                    land = (pose_landmarks[23] + pose_landmarks[24]) * 0.5
                else:
                    land = pose_landmarks[idx]
                idx_points[idx] = land

    
            for idx, po in idx_points.items():
                p_c = point_c[idx]
                cv2.circle(pro_frame, (int(po[0]), int(po[1])), 3, (int(p_c[0]), int(p_c[1]), int(p_c[2])), thickness=2)
    
    
            for liid in range(len(require_lines)):
                l_c = line_c[liid]
                l_start, l_end = idx_points[require_lines[liid][0]], idx_points[require_lines[liid][1]]
                cv2.line(pro_frame, (int(l_start[0]), int(l_start[1])), (int(l_end[0]), int(l_end[1])), (int(l_c[0]), int(l_c[1]), int(l_c[2])), thickness=2)

            
        output_frame = cv2.cvtColor(pro_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow(nameW, output_frame)
        out_video.write(output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break






