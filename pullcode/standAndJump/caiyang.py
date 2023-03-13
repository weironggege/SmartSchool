import cv2
import os

if __name__ == "__main__":

    cap = cv2.VideoCapture("jump4.mp4")
    
    i = 0
    while True:

        sucess, inpfr = cap.read()

        if not sucess:
            print("Error")
            break

        cv2.imwrite(os.path.join('./targetimgs', 'img_' + str(i) + ".jpg"), inpfr)

        i += 1

        if cv2.waitKey(1) in [ord('q'), 27]:
            break
