import cv2
import torch
import os
# from detect_person import detectp_box
# from detect_rope import detectropebox

if __name__ == "__main__":
    
    url = "rtsp://admin:yskj12345@192.168.2.203"

    cap = cv2.VideoCapture(url)
    
    model1 = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt") 
    model2 = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    model1.conf = .7
    model2.conf = .7
    
    cv2.namedWindow("detect ropeskip", cv2.WINDOW_NORMAL)
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # out_video = cv2.VideoWriter('./videores/detectropeskipv2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))



    n_repeat, flag = 0, False
    while True:

        success, inp_frame = cap.read()

        if not success:
            print("Error")
            break

        inp_frame = cv2.cvtColor(inp_frame, cv2.COLOR_BGR2RGB)
        
        res1 = model1(inp_frame)
        res2 = model2(inp_frame)

        box1 = res1.pandas().xyxy[0].to_numpy()
 
        box2 = res2.pandas().xyxy[0].to_numpy()

        if len(box1) > 0 and len(box2) > 0:
            cv2.rectangle(inp_frame, (int(box1[0,0]), int(box1[0,1])), (int(box1[0,2]), int(box1[0,3])), (0,0,255), thickness=1, lineType=4)
            cv2.rectangle(inp_frame, (int(box2[0,0]), int(box2[0,1])), (int(box2[0,2]), int(box2[0,3])), (0,225,0), thickness=1, lineType=4)
            
            '''
            if flag == 0 and box1[0,1] < box2[0,1]:
                n_repeat += 1
                flag = 1

            if flag == 1 and box1[0,3] > box2[0,3]:
                flag = -1
            
            if flag == -1 and (box1[0,3] < box2[0, 3] and box1[0,1] > box2[0,1]):
                flag = 0
            
            '''
            if not flag:
                flag = box1[0,1] < box2[0,1]

            if flag and box1[0,3] > box2[0,3]:
                n_repeat += 1
                flag = False
            

        
        cv2.putText(inp_frame, 'Valid:' + str(n_repeat), (900, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), thickness=6)

        inp_frame = cv2.cvtColor(inp_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("detect ropeskip", inp_frame)
        # out_video.write(inp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            video_cap.release()
            break


        
