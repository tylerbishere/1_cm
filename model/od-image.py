import time
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import detect
import cv2
import numpy as np
import os

def main():

    print("...Loading model...")
    start_time = time.time()

    #storing labels from label text file
    label_path = "model/labels.txt"
    labels = dataset.read_label_file(label_path)
     

    #making interpreter for model file
    model_path = "model/model_quant_edgetpu.tflite"
    interpreter = edgetpu.make_interpreter(model_path, device="usb")

    total_time = time.time() - start_time
    print("Finished loading model. Time={}s".format(total_time))


    start_time2 = time.time()
    interpreter.allocate_tensors()


    #configuring input image
    allFiles = os.listdir(os.getcwd() + "/10ims")
    print(allFiles)
    image_names = [("10ims/"+f) for f in allFiles]
    print(image_names)
    writeImages = []

    #array for detections information
    detections = []
    for i in image_names:
        
        image = cv2.imread(i)
        writeImages.append(image)
        height = image.shape[0]
        width = image.shape[1]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (640,640))
        image_array = np.expand_dims(image_resized, axis=0)
        image_array = (np.float32(image_array)-127.5)/127.5


        common.set_input(interpreter, image_array)



        #running inference
        interpreter.invoke()

        #storing detection information
        d = detect.get_objects(interpreter, 0.5, (1.0, 1.0))
        detections.append(d)

    end_time = time.time() - start_time2
    print("Detections done: {} seconds".format(end_time))

    #drawing detections to original image
    count = -1
    for results in detections:
        count += 1
        for o in results:
            box = o.bbox
            xmin = int(max(1, box.xmin*(width/640)))
            ymin = int(max(1, box.ymin*(height/640)))
            xmax = int(min(width, box.xmax*(width/640)))
            ymax = int(min(height, box.ymax*(height/640)))
            cv2.rectangle(writeImages[count], (xmin,ymin), (xmax, ymax), (255,0,0), 2)
    
        cv2.putText(writeImages[count], "Num Detections: " + str(len(detections)), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)    
        cv2.imwrite("10imsresults/t" + str(count) + ".jpg", writeImages[count])
if __name__ == '__main__':
    main()





