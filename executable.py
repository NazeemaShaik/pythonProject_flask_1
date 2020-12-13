async def read_image(img_path):
    import random
    import string
    import math
    import itertools
    import os

    import numpy as np
    import imgaug
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import sklearn.model_selection

    import keras_ocr
# recognizer = keras_ocr.recognition.Recognizer()
# recognizer.compile()
# recognizer.model.load_weights('recognizer_borndigital_5000.h5')
    import random
    import string
    import math
    import itertools
    import os

    import numpy as np
    import imgaug
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import sklearn.model_selection

    import keras_ocr

    recognizer = keras_ocr.recognition.Recognizer()
    recognizer.compile()
    recognizer.model.load_weights('recognizer_borndigital (1).h5')
    detector = keras_ocr.detection.Detector(weights='clovaai_general')
# from google.colab.patches import cv2_imshow
    import cv2
    import numpy as np
    import glob
    import random

# Load Yolo
    net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")  # give the path to cfg files

# Name custom object
    classes = ["Number Plate"]

# Images path
# v_image=input("enter path to image:")
    images_path = glob.glob(img_path)  # give the path to the image

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
    random.shuffle(images_path)
# loop through all the images
    for img_path in images_path:
    # Loading image
        img = cv2.imread(img_path)
        height, width, channels = img.shape

    # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

    # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                # Object detected
                    print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y), font, 2, color, 2)
    # cv2.imshow("vehicle_img",img)# to show the container alongwith the detected serial number
        img = img[y:y + h, x:x + w]
        cv2.imwrite("cropped_img.jpg", img)  # to save the image to the current directory
    # cv2.imshow("cropped_img",img)
    # predicted = recognizer.recognize(img)
    # print(f'Predicted: {(predicted).upper()}')
    # _ = plt.imshow(keras_ocr.tools.read(img))
        import cv2
        import numpy as np
        from scipy.ndimage import interpolation as inter


    # from google.colab.patches import cv2_imshow

        def correct_skew(image, delta=1, limit=5):
            def determine_score(arr, angle):
                data = inter.rotate(arr, angle, reshape=False, order=0)
                histogram = np.sum(data, axis=1)
                score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
                return histogram, score

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            scores = []
            angles = np.arange(-limit, limit + delta, delta)
            for angle in angles:
                histogram, score = determine_score(thresh, angle)
                scores.append(score)

            best_angle = angles[scores.index(max(scores))]

            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, best_angle - 20, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
                                 borderMode=cv2.BORDER_REPLICATE)

            return best_angle, rotated


        #if __name__ == '__main__':
        image = cv2.imread('cropped_img.jpg')
        angle, rotated = correct_skew(image)
            #print(angle)
        # cv2.imshow(rotated)
            #cv2.imwrite("cropped_img.jpg", img)
        cv2.imwrite("rotated.jpg", rotated)
        pipeline = keras_ocr.pipeline.Pipeline(detector=detector, recognizer=recognizer)
    # image, lines = next(image_generators[0])
        image = keras_ocr.tools.read('rotated.jpg')
        predictions = pipeline.recognize(images=[image])[0]
    # boxes = detector.detect(images=[image])[0]
        drawn = keras_ocr.tools.drawBoxes(
        image=image, boxes=predictions, boxes_format='predictions'
        )
        predicted=''
        for text, box in predictions:
            predicted=predicted+text

        #Predicted = [text for text, box in predictions]
        print(

        'Predicted:', [text for text, box in predictions])
    return predicted.upper()
    # plt.imshow(drawn)