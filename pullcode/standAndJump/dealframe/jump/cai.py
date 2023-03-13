import cv2


if __name__ == "__main__":

    cap = cv2.VideoCapture("./jump_out_caiyang.mp4")
    frame_c = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    while True:
        sucess, frame  = cap.read()

        if not sucess:
            print("Error")
            break

        if i == 0:
            cv2.imwrite("./s_fr.jpg", frame)

        if i == frame_c - 1:
            cv2.imwrite("./e_fr.jpg", frame)
    
        

        i += 1
        if cv2.waitKey(1) in [ord('q'), 27]:
            break
