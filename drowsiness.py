
import cv2 as cv
import dlib
from scipy.spatial import distance

def aspect_ratio(eye):
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2],eye[4])
    C = distance.euclidean(eye[0],eye[3])
    eye_aspect_ratio = (A+B)/(2.0*C)
    return eye_aspect_ratio

cap = cv.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


while True:
    _,frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray,face)
        lefteye = []
        righteye = []
        for n in range(36,42):
            x1 = landmarks.part(n).x
            y1 = landmarks.part(n).y
            lefteye.append((x1,y1))
            next_pt = n+1
            if n == 41:
                next_pt = 36
            x2 = landmarks.part(next_pt).x
            y2 = landmarks.part(next_pt).y
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)
        for n in range(42,48):
            x1 = landmarks.part(n).x
            y1 = landmarks.part(n).y
            righteye.append((x1,y1))
            next_pt = n+1
            if n == 47:
                next_pt = 42
            x2 = landmarks.part(next_pt).x
            y2 = landmarks.part(next_pt).y
            cv.line(frame,(x1,y1),(x2,y2),(0,255,0),1)

        left_eye = aspect_ratio(lefteye)
        right_eye = aspect_ratio(righteye)
        eye = (left_eye+right_eye)/2
        eye = round(eye,2)

        if eye < 0.26:
            cv.putText(frame,"DOZING OFF",(20,100),cv.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
            cv.putText(frame,"GET BACK TO YOUR SENSES",(20,400),cv.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
    cv.imshow('Drowsiness Detector',frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
        