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
    
    cap = cv2.VideoCapture("ysqiangpao.mp4")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    

    
    
    out_video = cv2.VideoWriter('./ysqiangpao_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))


    model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt") 
    model.conf = 0.5
    
    cv2.namedWindow("detect foot", cv2.WINDOW_NORMAL)
    
    k = 0
    while True:

        success, inp_frame = cap.read()

        if not success:
            print("Error")
            break

        inp_frame = cv2.cvtColor(inp_frame, cv2.COLOR_BGR2RGB)

        results = model(inp_frame)

        
        boxes = results.pandas().xyxy[0].to_numpy()
        
        # cv2.line(inp_frame, (517,986), (716,1230), (255,0,0), thickness=2)
        # print(boxes)
        if len(boxes) > 0:
             for i in range(len(boxes)):
                cv2.rectangle(inp_frame, (int(boxes[i,0]), int(boxes[i,1])), (int(boxes[i,2]), int(boxes[i,3])), (0,0,255), thickness=2, lineType=4)
                # print((boxes[i,0], boxes[i,1]), (boxes[i,2], boxes[i,3]))    
        
        if k >= 98:
            cv2.putText(inp_frame, "3 runway run in", (50, 750), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=6)



        inp_frame = cv2.cvtColor(inp_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("detect ropeskip", inp_frame)
        out_video.write(inp_frame)

        k += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            video_cap.release()
            break
    

