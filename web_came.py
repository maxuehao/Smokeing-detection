import  dlib  
import cv2
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("/Users/m/downloads/caffe/python")
import caffe
import numpy as np

def lodel_class_model(net, img, mean):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(mean).mean(1).mean(1))
    out = net.forward_all(data=np.asarray([transformer.preprocess('data', img)]))
    feature =  out['softmax'][0].tolist()
    cls_index = feature.index(max(feature))
    return cls_index,max(feature)

def geteye_rect(frame,name, img):
    bgrImg = frame
    temp = 0
    if bgrImg is None:  
        return False  
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)  
    facesrect = face_detector(rgbImg, 1)  
    if len(facesrect) <=0:  
        return False    
    for k, d in enumerate(facesrect):  
        shape = landmark_predictor(rgbImg, d)
        ptl=shape.part(48)
        #print [ptl.x,ptl.y]
        ptr=shape.part(54)
        #print [ptr.x,ptr.y]
        w = int(1.5*(ptr.x - ptl.x))
        h = w 
        x = ptl.x
        y = ptl.y - int(0.4*h)
        roi = frame[y:(y + h), x-int(0.4*w):(x + w)]
        label , Confidence = lodel_class_model(smoke_net, roi, smoke_mean)
        if label == 0:
            cv2.putText(img, "smokeing", (4 * (x - int(0.4 * w)), 4 * y-10), 0, 1, (255, 0, 0), 2)
            cv2.rectangle(img, (4 * (x - int(0.4 * w)), 4 * y), (4 * (x + w), 4 * (y + h)), (0, 0, 255), 2)
        else:
            cv2.rectangle(img, (4 * (x - int(0.4 * w)), 4 * y), (4 * (x + w), 4 * (y + h)), (0, 255, 0), 2)
        print label , Confidence

face_detector=dlib.get_frontal_face_detector()
landmark_predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)
temp = 0
smoke_net = caffe.Net('smoke.prototxt', 'smoke.caffemodel', caffe.TEST)
smoke_mean = 'smoke.npy'

while True:
    ret, frame = cap.read()
    img = frame
    frame = cv2.resize(frame, (320,180), interpolation=cv2.INTER_AREA)
    geteye_rect(frame,temp, img)
    print "fps_num:" + str(temp)
    cv2.imshow("capture", img)
    temp = temp + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()      


