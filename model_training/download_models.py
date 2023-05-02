# Download the shape predictor file
import urllib.request

url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")

# Extract the shape predictor file
import bz2

with open("shape_predictor_68_face_landmarks.dat", "wb") as f:
    f.write(bz2.open("shape_predictor_68_face_landmarks.dat.bz2").read())

# Download the face recognition model file
url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
urllib.request.urlretrieve(url, "dlib_face_recognition_resnet_model_v1.dat.bz2")

# Extract the face recognition model file
with open("dlib_face_recognition_resnet_model_v1.dat", "wb") as f:
    f.write(bz2.open("dlib_face_recognition_resnet_model_v1.dat.bz2").read())
