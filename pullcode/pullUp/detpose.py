from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import cv2
import numpy as np

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



if __name__ == "__main__":

    video_cap = cv2.VideoCapture("./yolopullUp/data/videos/yinti6.MOV")
    

    nameW = 'pose show'
    cv2.namedWindow(nameW, cv2.WINDOW_NORMAL)


    pose_tracker = mp_pose.Pose()
    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = './yinti6out.mp4' 

    out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps,
                            (video_width, video_height))
    

    n_valid, n_invalid, flag = 0, 0, 1
    while True:

        sucess, frame = video_cap.read()

        if not sucess:
            print("Error")
            break

        inp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = pose_tracker.process(image=inp_frame)


        pose_landmarks = result.pose_landmarks

        output_frame = inp_frame.copy()
        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)

            pose_landmarks = np.array(
                                        [[lmk.x * video_width, lmk.y * video_height, lmk.z * video_width]
                                         for lmk in pose_landmarks.landmark])

            pose_landmarks = pose_landmarks[:, :2]
            
            person_nose = pose_landmarks[landmark_names.index('nose')]
            
            person_lwrist, person_rwrist = pose_landmarks[landmark_names.index('left_wrist')], pose_landmarks[landmark_names.index('right_wrist')]
            
            cv2.circle(output_frame, (int(person_rwrist[0]), int(person_rwrist[1])-10), 5, (255,0,0), thickness=2)      
            

            if flag == 1 and person_nose[1] < 480:
                n_invalid += 1
                flag = 0
            elif (flag == 2 or flag == 3) and person_nose[1] < 480:
                n_invalid += 1
                flag = 0

            if person_nose[1] >= 480 and flag == 0:
                flag = 2
            '''        
            if person_nose[1] >= 550 and flag == 2:
                flag = 1
        
            if int(person_rwrist[0]) < 1080:
                if output_frame[int(person_rwrist[0]),int(person_rwrist[1])-10][2] < 100:
                    flag = 3
            ''' 

        # cv2.line(output_frame, (650,480), (1250, 480), (255,0,0), thickness=2, lineType=4)

        cv2.putText(output_frame, 'Valid:' + str(n_valid), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 4)
        cv2.putText(output_frame, 'InValid:' + str(n_invalid), (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 4)
        
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow(nameW, output_frame)
        out_video.write(output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break





