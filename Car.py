import cv2

#paths
img_file= "car2.jpg"
video = cv2.VideoCapture("carvideo.mp4")
classifier_file='cars.xml'

img = cv2.imread(img_file)
bnw= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#classify
car_tracker = cv2.CascadeClassifier(classifier_file)
cars= car_tracker.detectMultiScale(bnw)

while True:
    readSuccess, frame= video.read()

    if readSuccess:
        bwframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        videocars= car_tracker.detectMultiScale(bwframe)
        print(videocars)
        for (x,y,w,h) in videocars:
            cv2.rectangle(frame,(x,y),(x+w, y+h), (299,66,25), 2)

    else:
        break
    cv2.imshow('test 2', frame)
    cv2.waitKey(1)



# cv2.imshow('test',img) 
# cv2.waitKey()


print('success')