from mediapipe.python.solutions import pose as mp_pose
import cv2
import numpy as np
import math
import random


def _get_angle(vec1, vec2):
    v_prod = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    l_prod = math.sqrt((vec1[0]*vec1[0] + vec1[1]*vec1[1]) * (vec2[0]*vec2[0] + vec2[1]*vec2[1]))
    return math.degrees(math.acos(v_prod / l_prod))



if __name__ == "__main__":
    
    # url = "rtsp://admin:yskj12345@192.168.2.206"
   

    nameW = "pose imshow"

    cors = []
    for _ in range(5):
        cors.append(random.sample(list(np.arange(256)), 3))
    
    

    
    inp_frame = cv2.imread("./s_fr.jpg")

    pose_tracker = mp_pose.Pose()
    
    pro_frame = cv2.cvtColor(inp_frame, cv2.COLOR_BGR2RGB)

    result = pose_tracker.process(image=pro_frame)

    pose_landmarks = result.pose_landmarks

    if pose_landmarks is not None:

        pose_landmarks = np.array(
                                [[lmk.x * inp_frame.shape[1], lmk.y * inp_frame.shape[0], lmk.z * inp_frame.shape[1]]
                                for lmk in pose_landmarks.landmark])

        pose_landmarks = pose_landmarks[:, :2]

        # right_h, right_k, right_a = pose_landmarks[24], pose_landmarks[26], pose_landmarks[28]
        
        # cv2.circle(pro_frame, (int(right_h[0]), int(right_h[1])), 6, (int(cors[0][0]),int(cors[0][1]),int(cors[0][2])), -1)
        # cv2.circle(pro_frame, (int(right_k[0]), int(right_k[1])), 6, (int(cors[1][0]),int(cors[1][1]),int(cors[1][2])), -1)
        # cv2.circle(pro_frame, (int(right_a[0]), int(right_a[1])), 6, (int(cors[2][0]),int(cors[2][1]),int(cors[2][2])), -1)
        # cv2.line(pro_frame, (int(right_h[0]), int(right_h[1])), (int(right_k[0]), int(right_k[1])), (int(cors[3][0]),int(cors[3][1]),int(cors[3][2])), thickness=2)
        # cv2.line(pro_frame, (int(right_k[0]), int(right_k[1])), (int(right_a[0]), int(right_a[1])), (int(cors[4][0]),int(cors[4][1]),int(cors[4][2])), thickness=2)
        right_s, right_e = pose_landmarks[12], pose_landmarks[14]
        vc1, vc2 = (right_e[0]-right_s[0], right_e[1]-right_s[1]), (0, 1)
        print(_get_angle(vc1, vc2))
               
        output_frame = cv2.cvtColor(pro_frame, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("./poseana/quxijiao.jpg", output_frame)
        cv2.imshow(nameW, output_frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



