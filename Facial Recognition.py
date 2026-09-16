import cv2
import threading
import face_recognition
import pickle
import os

faceImg = None
DETECTION_SCALE  = 0
DETECTION_NEIGHBORS = 0
DETECTION_ACCURACY_TOLERANCE = 0
DISPLAY_SCALE = 0

def loadDatabase():
    if not os.path.exists("faces.pkl"):
        with open('faces.pkl', 'wb') as f: 
                    pickle.dump(([], []), f)
    with open("faces.pkl", "rb") as f:
        return pickle.load(f)
    

def loadDetector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

def update_camera():
    global latest_frame

    while running:
        ret, frame = cap.read()
        if ret:
            latest_frame = frame

def startCamera():
    cameraThread = threading.Thread(
        target=update_camera
    )

    cameraThread.start()
    return cameraThread


def resizeImg(img, scale):
    h, w = img.shape[:2]

    return cv2.resize(
        img,
        (w//scale, h//scale) #Floor Division
    )

def greyscaleImg(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def displayImg(img):
    resized = resizeImg(img, DISPLAY_SCALE)
    cv2.imshow("Facial Detection", resized)

def detectFaces(img):
    faces = faceCascade.detectMultiScale(
        img, #Image 
        1.05, # Scale Factor
        DETECTION_NEIGHBORS # Minimum # of Neighbors
    )
    return faces

def compareFace(face, database):
    #Convert colors
    rgbFace = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    #Encode Face
    encodings = face_recognition.face_encodings(rgbFace)

    if len(encodings) > 0:
        results = face_recognition.compare_faces(
            database, #Database
            encodings[0], #Encoded Face 
            tolerance=DETECTION_ACCURACY_TOLERANCE #Accuracy
        )

        #If a face is found, return the name
        if True in results:
            index = results.index(True)
            currentName = knownNames[index]
            return currentName
        else:
            return "Unknown"
        

def drawFaces(img, database, faces):
    for (x,y,fw,fh) in faces:
        x *= DETECTION_SCALE 
        y *= DETECTION_SCALE
        fw *= DETECTION_SCALE
        fh *= DETECTION_SCALE

        #SLICE 2D Array
        global faceImg
        faceImg = img[
            y:y+fh,
            x:x+fw
        ]

        #If a face is found
        if faceImg.size != 0:
            currentName = compareFace(faceImg, database)
            #Draw Rectangle Around Face
            cv2.rectangle(
                img, #Image
                (x,y), #Top Left
                (x+fw,y+fh), #Bottom Right
                (0,0,255), #Color
                3 #Thickness
            )

            #Add Name Text
            cv2.putText(
                img, #Image
                currentName, #Text
                (x,y-10), #Location
                cv2.FONT_HERSHEY_SIMPLEX, #Font
                1, #Font Scale
                (0,0,255), #Color
                2 #Thickness
            )

def registerFace():
    if faceImg is None:
        return

    name = input("Enter person's name: ")

    rgbFace = cv2.cvtColor(
        faceImg,
        cv2.COLOR_BGR2RGB
    )

    locations = face_recognition.face_locations(rgbFace)
    encodings = face_recognition.face_encodings(rgbFace, known_face_locations=locations)

    if len(encodings) > 0:
        knownEncodings.append(encodings[0])

        knownNames.append(name)


        with open("faces.pkl", "wb") as f:
            pickle.dump((knownEncodings, knownNames), f)

        print(name, "registered!")

DETECTION_SCALE = 1
DETECTION_NEIGHBORS = 3
DETECTION_ACCURACY_TOLERANCE = 0.5
DISPLAY_SCALE = 2

latest_frame = None
running = True

url = "http://10.228.187.110:8080/video"
cap = cv2.VideoCapture(url)

knownEncodings, knownNames = loadDatabase()
cameraThread = startCamera()
faceCascade = loadDetector()

while True:
    if latest_frame is None:
            continue

    img = latest_frame.copy()
    #scaledImg = resizeImg(img, DETECTION_SCALE) #Not Needed
    greyImg = greyscaleImg(img)
    faces = detectFaces(greyImg)
    drawFaces(img, knownEncodings, faces)

    # Create Window
    displayImg(img)

    #If ESC Pressed, End Program
    key = cv2.waitKey(1)
    if key == 27:
        break

    if key == ord("r"):
        registerFace()


running = False
cameraThread.join()
cap.release()
cv2.destroyAllWindows()
