import cv2

if __name__ == "__main__":

    video_cap = cv2.VideoCapture("../jump4.mp4")


    nameW = 'pose show'
    cv2.namedWindow(nameW, cv2.WINDOW_NORMAL)


    video_fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_video_path = './jump_outte.mp4'

    out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps,
                           (video_width, video_height))

    i = 0
    while True:

        sucess, frame = video_cap.read()

        if not sucess:
            print("Error")
            break

        if i >= 50 and i <= 76:
            out_video.write(frame)
            if i == 50:
                cv2.imwrite("./s_fr.jpg", frame)
            if i == 76:
                cv2.imwrite("./e_fr.jpg", frame)
        
        i += 1
        if cv2.waitKey(1) in [ord('q'), 27]:
            break
