import cv2

BLUR_RADIUS = 21
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))

cap = cv2.VideoCapture(0)

for i in range(10):
    success, frame = cap.read()
    if not success:
        exit(1)
gray_background = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
gray_background = cv2.GaussianBlur(gray_background,(BLUR_RADIUS,BLUR_RADIUS),0)

while True:
    success, frame = cap.read()
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame,(BLUR_RADIUS,BLUR_RADIUS),0)


    diff = cv2.absdiff(gray_background,gray_frame)   # todo 差值图像（原始图像和当前帧之间的插值）
    _, thresh = cv2.threshold(diff,40,255,cv2.THRESH_BINARY)   # 阈值图像
    ## todo 对图片进行腐蚀和扩张
    cv2.erode(thresh,erode_kernel,thresh,iterations=2)
    cv2.dilate(thresh,dilate_kernel,thresh,iterations=2)

    contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > 5000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)

    cv2.imshow('diff',diff)
    cv2.imshow('thresh',thresh)
    cv2.imshow('detection',frame)


    if cv2.waitKey(1) == 27:
        break