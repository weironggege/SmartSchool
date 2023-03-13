from mediapipe.python.solutions import pose as mp_pose
import cv2
import numpy as np
import math

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

def _get_angle(vec1, vec2):
    v_prod = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    l_prod = math.sqrt((vec1[0]*vec1[0] + vec1[1]*vec1[1]) * (vec2[0]*vec2[0] + vec2[1]*vec2[1]))
    return math.degrees(math.acos(v_prod / l_prod))


def _is_pan(lst):
    pr_p = lst[0]
    for i in range(1,len(lst)):
        cur_p = lst[i]
        if abs(cur_p[0] - pr_p[0]) > 10 or abs(cur_p[1] - pr_p[1]) > 10:
            return False
    return True


if __name__ == "__main__":
    
    # url = "rtsp://admin:yskj12345@192.168.2.206"
    url = "../jump4.mp4"
    video_cap = cv2.VideoCapture(url)


    nameW = 'pose show'
    cv2.namedWindow(nameW, cv2.WINDOW_NORMAL)


    pose_tracker = mp_pose.Pose()
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = './jump_out.mp4' 

    out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps,
                           (video_width, video_height))






    pose_tracker = mp_pose.Pose()
    
    tem_o = [(0.,0.)] * 6

    i, s_idx, e_idx = 0, 0, 0
    isstart, isend = True, True
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

            r_s_p, r_l_p, n_p, h_p = pose_landmarks[12], pose_landmarks[14], (pose_landmarks[11] +  pose_landmarks[12]) * 0.5, (pose_landmarks[23] +  pose_landmarks[24]) * 0.5

            vic1, vic2, vic3 = (r_l_p[0]-r_s_p[0], r_l_p[1]-r_s_p[1]), (h_p[0]-n_p[0], h_p[1]-n_p[1]), (0, 1)

            ang1, ang2 = _get_angle(vic1, vic2), _get_angle(vic2, vic3)
            
            if (ang1 > 20.0 and ang2 > 20.0) and isstart:
                s_idx = i
                isstart = False

            
            r_h = pose_landmarks[30]

            del tem_o[0]

            tem_o.append(r_h)

            if (_is_pan(tem_o) and tem_o[0][0] > 500) and isend:
                e_idx = i - 5
                isend = False
    
            
        output_frame = cv2.cvtColor(pro_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow(nameW, output_frame)
        out_video.write(output_frame)

        i += 1
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    print(s_idx, e_idx)



