import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from mediapipe.python.solutions import pose as mp_pose

import cv2


if __name__ == "__main__":
    inp_frame = cv2.imread("./s_fr.jpg")
    src_frame = cv2.imread("../../targetimgs/img_68.jpg")
    
    pose_tracker = mp_pose.Pose()

    pro_frame = cv2.cvtColor(inp_frame, cv2.COLOR_BGR2RGB)

    result = pose_tracker.process(image=pro_frame)

    pose_landmarks = result.pose_landmarks

    if pose_landmarks is not None:

        pose_landmarks = np.array(
                                [[lmk.x * inp_frame.shape[1], lmk.y * inp_frame.shape[0], lmk.z * inp_frame.shape[1]]
                                for lmk in pose_landmarks.landmark])

        pose_landmarks = pose_landmarks[:, :2]

        right_h = pose_landmarks[30]



    pos = []
    with open("coor.txt", "r") as fr:
        for li in fr.readlines():
            pt = li.strip().split(" ")
            pos.append((round(float(pt[0]), 2), round(-float(pt[1]), 2)))
   

    x = [r[0] for r in pos]
    y = [r[1] for r in pos]
    f1 = np.polyfit(x, y, 3)
    yvals=np.polyval(f1, x)  
    x_new = np.linspace(min(x), max(x), 300) 
    y_smoth = make_interp_spline(x, yvals)(x_new)
  
    x_new_tem = [(t * 5 + right_h[0]) for t in x_new]
    y_smath_tem = [(right_h[1] - t * 5) for t in y_smoth]
    
    for i in range(len(x_new_tem)-1):
        cv2.line(src_frame, (int(x_new_tem[i]), int(y_smath_tem[i])), (int(x_new_tem[i+1]), int(y_smath_tem[i+1])), (255,0,0), thickness=2)
    

    wname = "pose"
    cv2.imwrite("./max_fr_tem.jpg", src_frame)
    # cv2.imshow(wname, inp_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    
    
    '''
    plt.figure(figsize=(20,5))
    plt.scatter(x, y, s=5, c='b')

    plt.plot(x_new, y_smoth, c='r', label='fitting values')
    plt.legend(loc='best')
    plt.savefig("./nihe.jpg")
    plt.show()
    '''
