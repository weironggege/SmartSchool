import cv2
import torch
import os

if __name__ == "__main__":
    
    
    srcimg = cv2.imread("img_0.jpg")
    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    model.conf = .6
     
    res_di = [None,None,6,16,5,15,13,4,12,None,8,7,14]

    res = model(srcimg)
    
    boxes = res.pandas().xyxy[0].to_numpy()
    for idb in range(len(boxes)):
        box = boxes[idb]
        cv2.rectangle(srcimg, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(srcimg, str(res_di[idb]), (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    wname = "detect ocr"

            
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)

    cv2.imshow(wname, srcimg)
    cv2.imwrite("imgout.jpg", srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
            
        




        
