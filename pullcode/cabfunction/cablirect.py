import cv2
import numpy as np

x1, y1 = 0, 0 
y2, y2 = 0, 0 
x3, y3 = 0, 0
x4, y4 = 0, 0
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
    global x1, y1, x2, y2, x3, y3, x4, y4, drawing, flag
    if event == cv2.EVENT_LBUTTONDOWN:
        if x < x1+5 and x > x1-5 and y < y1+5 and y > y1-5:
            drawing = True
            flag = 0
            print("click")
        elif x < x2+5 and x > x2-5 and y < y2+5 and y > y2-5:
            drawing = True
            flag = 1
            print("click")
        elif x < x3+5 and x > x3-5 and y < y3+5 and y > y3-5:
            drawing = True
            flag = 2
            print("click")
        elif x < x4+5 and x > x4-5 and y < y4+5 and y > y4-5:
            drawing = True
            flag = 3
            print("click")
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing:
            if flag == 0:
                x1 = x
                y1 = y
            elif flag == 1:
                x2 = x
                y2 = y
            elif flag == 2:
                x3 = x
                y3 = y
            elif flag == 3:
                x4 = x
                y4 = y

            canvas[:, :, :] = img[:, :, :]
            for pt in [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]:
                cv2.circle(canvas, pt, 3, (0, 255, 0), -1)
            cv2.line(canvas, (x1,y1), (x2,y2), (255,0,0), thickness=2)
            cv2.line(canvas, (x2,y2), (x3,y3), (255,0,0), thickness=2)
            cv2.line(canvas, (x3,y3), (x4,y4), (255,0,0), thickness=2)
            cv2.line(canvas, (x4,y4), (x1,y1), (255,0,0), thickness=2)

    if event == cv2.EVENT_LBUTTONUP:
        if drawing:
            if flag == 0:
                x1 = x
                y1 = y
            elif flag == 1:
                x2 = x
                y2 = y
            elif flag == 2:
                x3 = x
                y3 = y
            elif flag == 3:
                x4 = x
                y4 = y

            canvas[:, :, :] = img[:, :, :]
            for pt in [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]:
                cv2.circle(canvas, pt, 3, (0, 255, 0), -1)
            cv2.line(canvas, (x1,y1), (x2,y2), (255,0,0), thickness=2)
            cv2.line(canvas, (x2,y2), (x3,y3), (255,0,0), thickness=2)
            cv2.line(canvas, (x3,y3), (x4,y4), (255,0,0), thickness=2)
            cv2.line(canvas, (x4,y4), (x1,y1), (255,0,0), thickness=2)
        drawing = False
        flag = -1

if __name__ == "__main__":
    url = "rtsp://admin:yskj12345@192.168.2.206"

    cap = cv2.VideoCapture(url)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = cv2.VideoWriter('./cablirectout.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))


    cv2.namedWindow('Mouse Response', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Mouse Response', mouse_drawing)
    
    x1, y1 = 300, 300
    x2, y2 = 700, 300
    x3, y3 = 700, 700
    x4, y4 = 300, 700
    
    while True:
        sucess, canvas = cap.read()


        if not sucess:
            print("Error")
            break
        
       


        # canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)
        # canvas = cv.imread('images/2.png', cv.IMREAD_COLOR)

        cv2.line(canvas, (x1,y1), (x2,y2), (255,0,0), thickness=2)
        cv2.line(canvas, (x2,y2), (x3,y3), (255,0,0), thickness=2)
        cv2.line(canvas, (x3,y3), (x4,y4), (255,0,0), thickness=2)
        cv2.line(canvas, (x4,y4), (x1,y1), (255,0,0), thickness=2)

        for pt in [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]:
            cv2.circle(canvas, pt, 3, (0, 255, 0), -1)
        
        cv2.imshow("Mouse Response", canvas)
        out_video.write(canvas)

        if cv2.waitKey(1) == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
