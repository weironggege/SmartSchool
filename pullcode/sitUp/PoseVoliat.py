import math
import numpy as np
import datetime
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import cv2


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


class PoseVoliation(object):

    def __int__(self):
        pass


    def _get_angle(self, p_start, p_middle, p_end):
        """
        :param p_start:
        :param p_middle:
        :param p_end:
        :return: the angle of two vector
        """
        p1, p2 = [(p_middle[i] - p_start[i]) for i in range(2)], [(p_end[i] - p_middle[i]) for i in range(2)]
        v_prod = p1[0] * p2[0] + p1[1] * p2[1]
        l_prod = math.sqrt((p1[0]*p1[0] + p1[1]*p1[1]) * (p2[0]*p2[0] + p2[1]*p2[1]))
        return math.degrees(math.acos(v_prod / l_prod))



    def _get_dis(self, p_start, p_end):
        p = [(p_end[i] - p_start[i]) for i in range(2)]
        return math.sqrt(p[0]*p[0] + p[1]*p[1])


    def _pv_leg_angle(self, landmarks, landmark_names):
        right_hip = landmarks[landmark_names.index('right_hip')]
        right_knee = landmarks[landmark_names.index('right_knee')]
        right_ankle = landmarks[landmark_names.index('right_ankle')]
        return 180 - self._get_angle(right_hip, right_knee, right_ankle)


    def _pv_handtohead_dis(self, landmarks, landmark_names):
        right_wrist = landmarks[landmark_names.index('right_wrist')]
        nose = landmarks[landmark_names.index('nose')]
        return self._get_dis(right_wrist, nose)

    def _pv_elbowtoknee_dis(self, landmarks, landmark_names):
        right_elbow = landmarks[landmark_names.index('right_elbow')]
        right_knee = landmarks[landmark_names.index('right_knee')]
        return self._get_dis(right_elbow, right_knee)

    def _pv_should_angle(self, landmarks, landmark_names):
        right_should = landmarks[landmark_names.index('right_elbow')]
        right_hip = landmarks[landmark_names.index('right_hip')]
        p1, p2 = [(right_hip[i] - right_should[i]) for i in range(2)], [100.0, 0]
        v_prod = p1[0] * p2[0] + p1[1] * p2[1]
        l_prod = math.sqrt(
            (p1[0] * p1[0] + p1[1] * p1[1]) * (p2[0] * p2[0] + p2[1] * p2[1]))
        return math.degrees(math.acos(v_prod / l_prod))

    def _no_pv_dect(self, landmarks, landmark_names):
        hthd = self._pv_handtohead_dis(landmarks, landmark_names)
        lta = self._pv_leg_angle(landmarks, landmark_names)
        return hthd < 80.0 and lta < 140.0


    def _show_pv_leg_angle(self, input_video_path):
        video_cap = cv2.VideoCapture(input_video_path)

        pose_tracker = mp_pose.Pose()
        video_fps = video_cap.get(cv2.CAP_PROP_FPS)
        video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_win = "mediapose"
        cv2.namedWindow(out_win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(out_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


        while True:

            success, input_frame = video_cap.read()

            if not success:
                break
            #
            now_time = datetime.datetime.now()

            timestr = '_'.join([str(now_time.month), str(now_time.day), str(now_time.hour), str(now_time.minute), str(now_time.second)])



            if cv2.waitKey(1) & 0xFF == ord('s'):
                output_video_path = '../' + timestr + '@ceshi.mp4'

                out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps,
                                        (video_width, video_height))

                n_count, p_count, flag = 0, 0, False

                while True:

                    success, input_frame = video_cap.read()

                    if not success:
                        break


                    input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

                    result = pose_tracker.process(image=input_frame)


                    pose_landmarks = result.pose_landmarks

                    output_frame = input_frame.copy()
                    # if pose_landmarks is not None:
                    #     mp_drawing.draw_landmarks(
                    #         image=output_frame,
                    #         landmark_list=pose_landmarks,
                    #         connections=mp_pose.POSE_CONNECTIONS)

                    if pose_landmarks is not None:

                        pose_landmarks = np.array(
                                        [[lmk.x * video_width, lmk.y * video_height, lmk.z * video_width]
                                         for lmk in pose_landmarks.landmark])

                        pose_landmarks = pose_landmarks[:, :2]

                        # val = round(self._pv_leg_angle(pose_landmarks, landmark_names), 2)
                        # val = round(self._pv_handtohead_dis(pose_landmarks, landmark_names), 2)
                        # val = round(self._pv_elbowtoknee_dis(pose_landmarks, landmark_names), 2)
                        val = round(self._pv_should_angle(pose_landmarks, landmark_names), 2)

                        if not flag:
                            flag = val > 120.0

                        if val < 20.0 and flag:
                            if self._no_pv_dect(pose_landmarks, landmark_names):
                                n_count += 1
                            else:
                                p_count += 1
                            flag = False

                        # if val < 80:
                        #     res = "Hold Head"
                        # else:
                        #     res = "No Hold Head"


                        cv2.putText(output_frame, 'Valid:' + str(n_count), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 4)
                        cv2.putText(output_frame, 'Invalid:' + str(p_count), (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 4)
                    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

                    cv2.imshow(out_win, output_frame)
                    out_video.write(output_frame)

                    if cv2.waitKey(1) & 0xFF == ord('e'):
                        break

            cv2.putText(input_frame, "Press S to start detect", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 4)
            cv2.putText(input_frame, "Press q to quit", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 4)
            cv2.imshow(out_win, input_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                video_cap.release()
                break


if __name__ == "__main__":
    pv = PoseVoliation()
    pv._show_pv_leg_angle(0)


