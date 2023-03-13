import cv2
import numpy as np

x1, y1 = 0, 0 
y2, y2 = 0, 0 
flag = -1
drawing = False
canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
img = np.copy(canvas)
# canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)
# canvas = cv.imread('images/2.png', cv.IMREAD_COLOR)
# img = np.copy(canvas)
# cv.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2, 8, 0)
# for pt in [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]:
#     cv.circle(canvas, pt, 3, (0, 255, 0), -1)


def mouse_drawing(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, flag
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < x1+10 and x > x1-10 and y < y1+10 and y > y1-10:
            drawing = True
            flag = 0
            print("click")
        elif x < x2+10 and x > x2-10 and y < y2+10 and y > y2-10:
            drawing = True
            flag = 1
            print("click")
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing:
            if flag == 0:
                x1 = x
                y1 = y
            elif flag == 1:
                x2 = x
                y2 = y

            canvas[:, :, :] = img[:, :, :]
            for pt in [(x1,y1),(x2,y2)]:
                cv2.circle(canvas, pt, 3, (0, 255, 0), -1)
            cv2.line(canvas, (x1,y1), (x2,y2), (255,0,0), thickness=2)

    if event == cv2.EVENT_LBUTTONUP:
        if drawing:
            if flag == 0:
                x1 = x
                y1 = y
            elif flag == 1:
                x2 = x
                y2 = y

            canvas[:, :, :] = img[:, :, :]
            for pt in [(x1,y1),(x2,y2)]:
                cv2.circle(canvas, pt, 3, (0, 255, 0), -1)
            cv2.line(canvas, (x1,y1), (x2,y2), (255,0,0), thickness=2)
        drawing = False
        flag = -1
        print([x1,y1,x2,y2])

if __name__ == "__main__":
    url = "rtsp://admin:yskj12345@192.168.2.206"

    cap = cv2.VideoCapture(url)
    
    cv2.namedWindow('Mouse Response', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Mouse Response', mouse_drawing)
    
    x1, y1 = 300, 300
    x2, y2 = 700, 300
    
    while True:
        sucess, canvas = cap.read()


        if not sucess:
            print("Error")
            break
        
       


        # canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)
        # canvas = cv.imread('images/2.png', cv.IMREAD_COLOR)

        # cv2.line(canvas, (x1,y1), (x2,y2), (255,0,0), thickness=2)
        cv2.line(canvas, (x1,y1), (x2,y2), (0,0,255), thickness=2)
        for pt in [(x1,y1),(x2,y2)]:
            cv2.circle(canvas, pt, 5, (255, 0, 0), -1)
        
        cv2.imshow("Mouse Response", canvas)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
