import torch
import cv2


def detectropebox(cv_frame):
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt")  # or yolov5n - yolov5x6, custom

    # Images
    # img = 'img_0.jpg'  # or file, Path, PIL, OpenCV, numpy, list

    # Inference
    results = model(cv_frame)

    # Results
    # print(results.pandas().xyxy)  # or .s
    x1, y1, x2, y2, = float(results.pandas().xyxy[0].xmin), float(results.pandas().xyxy[0].ymin), float(results.pandas().xyxy[0].xmax), float(results.pandas().xyxy[0].ymax)
    return [x1, y1, x2, y2]


if __name__ == "__main__":
    
    """
    img = cv2.imread("./img_0.jpg")
    bbox = detectropebox(img)

    for coor in bbox:
        print(coor)
    """

    model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt") 
    cap = cv2.VideoCapture("../../videos/ropeSkip1.mp4")
    model.conf = 0.2
    
    cv2.namedWindow("detect ropeskip", cv2.WINDOW_NORMAL)

    while True:

        success, inp_frame = cap.read()

        if not success:
            print("Error")
            break

        inp_frame = cv2.cvtColor(inp_frame, cv2.COLOR_BGR2RGB)

        results = model(inp_frame)

        
        box2 = results.pandas().xyxy[0].to_numpy()
        # print(box2[0])
        if len(box2) > 0:
             cv2.rectangle(inp_frame, (int(box2[0,0]), int(box2[0,1])), (int(box2[0,2]), int(box2[0,3])), (0,225,0), thickness=1, lineType=4)
             # print((box2[0,0], box2[0,1]), (box2[0,2], box2[0,3]))    
        inp_frame = cv2.cvtColor(inp_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("detect ropeskip", inp_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            video_cap.release()
            break


