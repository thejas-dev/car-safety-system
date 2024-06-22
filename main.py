import numpy as np
import imutils
import pickle
import time
import cv2
import csv
import os
from imutils import paths
import serial
import dlib
from sklearn.preprocessing import LabelEncoder
from scipy.spatial import distance as dist
from sklearn.svm import SVC
from imutils import face_utils
import pickle
import winsound
frequency = 2500
duration = 1000

def eyeAspectRatio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

count = 0
earThresh = 0.3 #distance between vertical eye coordinate Threshold
earFrames = 48 #consecutive frames for eye closure
shapePredictor = "shape_predictor_68_face_landmarks.dat"

detector3 = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

#get the coord of left & right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

serialPort = serial.Serial(port="COM8", baudrate=115200,
                    bytesize=8,timeout=1,stopbits=serial.STOPBITS_ONE)
dataset = 'dataset'
embeddingFile = "output/embeddings.pickle" #initial name for embedding file
embeddingModel = "openface_nn4.small2.v1.t7" #initializing model for embedding Pytorch

#initialization of caffe model for face detection
prototxt = "model/deploy.prototxt"
model =  "model/res10_300x300_ssd_iter_140000.caffemodel"

#loading caffe model for face detection
#detecting face from Image via Caffe deep learning
detector2 = cv2.dnn.readNetFromCaffe(prototxt, model)

#loading pytorch model file for extract facial embeddings
#extracting facial embeddings via deep learning feature extraction
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

#initialization
knownEmbeddings = []
knownNames = []
total = 0
conf = 0.5

#New & Empty at initial
recognizerFile = "output/recognizer.pickle"
labelEncFile = "output/le.pickle"

recognizer = pickle.loads(open(recognizerFile, "rb").read())
le = pickle.loads(open(labelEncFile, "rb").read())

cascade = 'haarcascade_frontalface_default.xml'
detector = cv2.CascadeClassifier(cascade)

box = []
start = False
nameArray = []
personInArray = []
personIdArray = []

personId=0
idObtained = False
j=0

cam = cv2.VideoCapture(0)

state = 1
name = ''
personIn = False
noPersonConfirm = False
drowsinessConfirm = False

def loadRecognizer():
    recognizer = pickle.loads(open(recognizerFile, "rb").read())
    le = pickle.loads(open(labelEncFile, "rb").read())

def createData():
    global state
    global idObtained
    idObtained = False
    state = 1
    Name = str(input("Enter your Name : "))
    Roll_Number = int(input("Enter your Number : "))
    sub_data = Name
    path = os.path.join(dataset, sub_data)

    if not os.path.isdir(path):
        os.mkdir(path)
        print(sub_data)

    info = [str(Name), str(Roll_Number)]
    with open('student.csv', 'a') as csvFile:
        write = csv.writer(csvFile)
        write.writerow(info)
    csvFile.close()

    print("Starting video stream...")
    time.sleep(2.0)
    total = 0
    while total < 10:
        print(total)
        _, frame = cam.read()
        img = imutils.resize(frame, width=400)
        rects = detector.detectMultiScale(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            p = os.path.sep.join([path, "{}.png".format(
                str(total).zfill(5))])
            cv2.imwrite(p, img)
            total += 1

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    #cam.release()
    total = 0
    time.sleep(2)
    imagePaths = list(paths.list_images(dataset))
    for (i, imagePath) in enumerate(imagePaths):
        print("Processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        #converting image to blob for dnn face detection
        imageBlob = cv2.dnn.blobFromImage(
        	cv2.resize(image, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)

        #setting input blob image
        detector2.setInput(imageBlob)
        #prediction the face
        detections = detector2.forward()

        if len(detections) > 0:
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > conf:
                #ROI range of interest
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                if fW < 20 or fH < 20:
                    continue
	    	#image to blob for face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
			#facial features embedder input image face blob
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    print("Embedding:{0} ".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    print(data)
    f = open(embeddingFile, "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("Process Completed")
    
    time.sleep(2)
    print("Loading face embeddings...")
    data = pickle.loads(open(embeddingFile, "rb").read())

    print("Encoding labels...")
    labelEnc = LabelEncoder()
    labels = labelEnc.fit_transform(data["names"])
    #state=1
    time.sleep(2.0)

    print("Training model...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    f = open(recognizerFile, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    f = open(labelEncFile, "wb")
    f.write(pickle.dumps(labelEnc))
    f.close()
    

def checkFace():
    global personIn
    global name
    global personId
    global idObtained
    global j
    global start
    global nameArray
    global personInArray
    global personIdArray
    global count
    global noPersonConfirm
    global drowsinessConfirm
    if(idObtained == False):
        personId = int(input("Enter your id :- "))
        idObtained = True      
    _, frame = cam.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector3(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        #cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        #cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

        if ear < earThresh:
            count += 1

            if count >= earFrames:
                winsound.Beep(frequency, duration)
                if(drowsinessConfirm == False):
                    serialPort.write(b"DR")
                    drowsinessConfirm = True
            else:
                if(drowsinessConfirm == True):
                    serialPort.write(b"NDR")
                    drowsinessConfirm = False
                
        else:
            count = 0
            
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
    cv2.imshow("Frame", frame)
    detector2.setInput(imageBlob)
    detections = detector2.forward()
    personInArray.append(detections[0][0][0][2])
    if(len(personInArray) > 20):
        val2 = 0
        for i in range(0,len(personInArray)):
            val2 = val2 + personInArray[i]
        #print(val2/len(personInArray))
        if (val2/len(personInArray) < 0.7):
            if(noPersonConfirm == False):
                personNotFound()
                noPersonConfirm = True
        else:
            noPersonConfirm = False
    #print(detections[0][0][0][2])
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf:
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue
            personIn = True
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            text = "{}  : {:.2f}%".format(name, proba * 100)
            if(start == True):
                personIdArray.append(j)
                if(len(personIdArray) > 15):
                    start = False
                    checkAndSend()
            #print(j)
            #y = startY - 10 if startY - 10 > 10 else startY + 10
            #cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            #cv2.putText(frame, text, (startX, y),
             #           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        cam.release()
        cv2.destroyAllWindows()
        state = 2

def personNotFound():
    serialPort.write(b"NP")
    print("No person")

def personFound():
    serialPort.write(b"OK")
    print("Person found")

def startDetecting():
    global start
    global nameArray
    global personInArray
    global personIdArray
    start = True
    nameArray = []
    personInArray = []
    personIdArray = []

def checkAndSend():
    val = 0
    #val2 = 0
    for i in range(0,len(personIdArray)):
        val = val + personIdArray[i]
    #for i in range(0,len(personInArray)):
    #    val2 = val2 + personInArray[i]
    #personin = True if val2/len(personInArray) > 0.5 else False
    personid = round(val/len(personIdArray))
    if(name == "UnAuthorised"):
        serialPort.write(b"Unauthorised")
        print(name)
    #elif(personin == False):
    #    serialPort.write(b"NoPerson")
    #    print(personIn)
    elif(personid != personId):
        serialPort.write(b"ID")
        print(personid)
    else:
        serialPort.write(b"OK")

while True:
    if(serialPort.in_waiting > 0):
        serialString = serialPort.readline()
        #print(serialString.decode('Ascii').strip())
        if(serialString.decode('Ascii').strip() == 'createNew' ):
            cv2.release()
            state=0
        if(serialString.decode('Ascii').strip() == 'Check Data' ):
            print("Im called")
            startDetecting()
        if(serialString.decode('Ascii').strip() == 'Change' ):
            state = 1
            print("State changed")
            serialPort.write(b"DONE")
    if(state == 0):
        createData()
    elif(state == 1):
        loadRecognizer()
        checkFace()
    #switch state:
     #   case 0:
      #      createData()
       # case 1:
        #    loadRecognizer()
         #   checkFace()
            
    
