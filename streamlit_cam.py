from cgitb import html
import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import logging as log
import datetime as dt
from time import sleep

cascade = 'facedetect.xml'
faceCascade = cv2.CascadeClassifier(cascade)
log.basicConfig(filename='webcam.log',level = log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

# Streamlit begins
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image : url("https://coolbackgrounds.io/images/backgrounds/index/compute-ea4c57a4.png") ;
        background-size:cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# CSS styles for Streamlit components
st.markdown(
    """
    <style>
    body {
        background-image: url('background.jpg');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center center;
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
    }
    .container {
        max-width: 800px;
        margin: 20px auto;
        padding: 20px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.2);
    }
    .header {
        background-color: #333;
        color: #fff;
        padding: 20px;
        text-align: center;
    }
    .header h1 {
        font-size: 36px;
        margin: 0;
        padding: 0;
    }
    .header p {
        font-size: 18px;
        margin: 10px 0;
        padding: 0;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)


# Information about face detection
st.markdown(
    """
    <div class="st-eb">
        <h2>Real-Time Face Detection using OpenCV</h2>
        <p>
            Face detection is a crucial technology in computer vision that involves identifying and locating human faces within images or video streams. It plays a significant role in various applications, including security, surveillance, photography, and augmented reality. Detecting faces in real-time using OpenCV provides the foundation for developing advanced applications that rely on facial recognition and analysis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)



face_image = np.array(cv2.imread('face.png'))
st.image(face_image, use_column_width=True,channels='BGR')  

st.header("Let's detect")


if st.button('detect'):
    while True:
        if not video_capture.isOpened():
            print('Unable to load camera')
            sleep(5)
            pass

        # capture frame by frame
        _,frame = video_capture.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30,30)

        )    

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        if anterior!=len(faces):
            anterior = len(faces)
            log.info('faces:' + str(len(faces)) + 'at' + str(dt.datetime.now()))
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()  


