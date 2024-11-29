import cv2
import streamlit as st
import numpy as np
from PIL import Image

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(r"D:\DATA ANALYSIS\4TH MONTH\notes\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r"D:\DATA ANALYSIS\4TH MONTH\notes\haarcascade_eye.xml")

# Streamlit app title
st.title(" Face & Eye Detection")

# Apply custom CSS for the background
bg_image_url = "https://t3.ftcdn.net/jpg/02/89/72/00/240_F_289720001_JeepgWkYfDB1u0lSvkkF12vRV22DTOLd.jpg"  # Replace with your image URL
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{bg_image_url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Option to select the image source
option = st.selectbox("Select Image Source", ("Upload an Image", "Click a Picture"))

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Select Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(100, 100)
        )
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Region of interest (ROI) for eyes within the detected face
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]

            # Detect eyes
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=10,
                minSize=(30, 30)
            )
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(roi_color, "Eye", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the output
        st.image(image, caption='Processed Image', use_column_width=True)

elif option == "Click a Picture":
    st.write("Click the button to capture a frame from the webcam")
    capture_button = st.button("Extract")

    if capture_button:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(100, 100)
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Region of interest (ROI) for eyes within the detected face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                # Detect eyes
                eyes = eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=10,
                    minSize=(30, 30)
                )
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    cv2.putText(roi_color, "Eye", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Release the capture and display the frame
            cap.release()
            st.image(frame, caption='Extract Image', use_column_width=True)
        else:
            st.write("Failed to capture from webcam")
