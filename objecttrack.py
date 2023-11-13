import cv2 as cv

tracker = cv.TrackerKCF_create()
cap = cv.VideoCapture(0)


while True:
    _,frame = cap.read()
    cv.imshow('frame',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

bbox = cv.selectROI(frame,False)

ok = tracker.init(frame,bbox)
cv.destroyWindow('ROI selector')

while True:
    ok,frame = cap.read()
    ok,bbox = tracker.update(frame)

    if ok:
        p1 = (int(bbox[0]),int(bbox[1]))
        p2 = (int(bbox[0]+bbox[2]),
        int(bbox[1]+bbox[3]))
        cv.rectangle(frame,p1,p2,(0,0,255),2,2)
    
    cv.imshow("Tracking",frame)
    if cv.waitKey(1) & 0xFF ==ord('q'):
        break

cap.release()
cv.destroyAllWindows()