import cv2


def detectp_box(cv_frame):
    # Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    model.conf = .8
    # Images
    # img = 'img_0.jpg'  # or file, Path, PIL, OpenCV, numpy, list

    # img = cv2.imread('../data/images/bus.jpg')[:,:,::-1]

    # Inference
    results = model(cv_frame)

    # Results
    # results.show()
    x1, y1, x2, y2, = float(results.pandas().xyxy[0].xmin), float(results.pandas().xyxy[0].ymin), float(results.pandas().xyxy[0].xmax), float(results.pandas().xyxy[0].ymax)
    return [x1, y1, x2, y2]

def zhouwei(i, j):
    allids = []
    for k in [i-1, i, i+1]:
        for j in [j-1, j, j+1]:
            allids.append([k,j])
    return allids


if __name__ == "__main__":
    
    namew = 'draw circle'
    cv2.namedWindow(namew, cv2.WINDOW_NORMAL)

    img = cv2.imread('img_0.jpg')
    for row,col in zhouwei(1110, 760):
        print(img[row, col])
    cv2.circle(img, (1110, 760), 10, (0,0,255), thickness=2)

    cv2.imshow(namew, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

