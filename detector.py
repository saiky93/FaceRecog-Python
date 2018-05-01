from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2,os
import base64
import numpy as np
from PIL import Image
import uuid

app = Flask(__name__)
CORS(app)

@app.route('/takePictures/<employeeID>', methods=['GET'])
def generateDataSet(employeeID):
    assert employeeID == request.view_args['employeeID']
    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier('Classifiers/face.xml')
    i = 0
    offset = 50
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            i = i + 1
            cv2.imwrite("dataSet/face-" + employeeID + '.' + str(i) + ".jpg",
                        gray[y - offset:y + h + offset, x - offset:x + w + offset])
            cv2.waitKey(100)
        if i > 19:
            cam.release()
            cv2.destroyAllWindows()
            break
    train()
    return jsonify({'status': 'generated'})

def train():
    recognizer = cv2.face.LBPHFaceRecognizer_create(threshold = 63)
    cascadePath = "Classifiers/face.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    path = 'dataSet'
    images, labels = get_images_and_labels(path,faceCascade)
    cv2.imshow('test', images[0])
    cv2.waitKey(1)

    recognizer.train(images, np.array(labels))
    recognizer.write('trainer/trainer.yml')
    cv2.destroyAllWindows()


def get_images_and_labels(path,faceCascade):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("face-", ""))

        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(10)
    # return the images list and labels list
    return images, labels


@app.route('/predict', methods=['POST'])
def get_predictPost():

    imageBase64 = request.get_json()['imguri'].replace("data:image/jpeg;base64,","")

    imgdata = base64.b64decode(imageBase64)
    uniqueName = str(uuid.uuid4())
    filename = "image"+uniqueName+".jpg"
    with open(filename, 'wb') as f:
        f.write(imgdata)
    threshold = 60
    recognizer = cv2.face.LBPHFaceRecognizer_create(threshold)
    recognizer.read('trainer/trainer.yml')
    cascadePath = "Classifiers/face.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);
    path = 'dataSet'
    # Read the image
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    predicted=-1
    for(x,y,w,h) in faces:
        label, conf = recognizer.predict(gray[y:y+h,x:x+w])
        predicted=label
        print("label : ",label,", conf : ",conf)
    try:
        os.remove(filename)
    except OSError:
        pass

    return jsonify({'employeeId': predicted})



if __name__ == '__main__':
    app.run(debug=True)