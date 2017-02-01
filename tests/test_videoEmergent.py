import cv2

print'***************************'
cap = cv2.VideoCapture('/media/lab/New Volume 2/idZebLib/TU20160920/37dpf/adapted_lowIR.avi')
'/media/lab/New Volume 2/idZebLib/TU20160920/36dpf/IR(half)/group_1.avi'
'/media/lab/New Volume/Tests Emergent Vision 20000HT/test5.avi'
print cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
print cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
print cap.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO)
print cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
print cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
print cap.get(cv2.cv.CV_CAP_PROP_FPS)
print cap.get(cv2.cv.CV_CAP_PROP_FOURCC)
print cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
print cap.get(cv2.cv.CV_CAP_PROP_FORMAT)

ret, frame = cap.read()
print frame
print ret


print'***************************'
cap = cv2.VideoCapture('/media/lab/New Volume/Tests Emergent Vision 20000HT/group_1_camera_1_20161028T082945_1.avi' )
print cap.get(cv2.cv.CV_CAP_PROP_POS_MSEC)
print cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
print cap.get(cv2.cv.CV_CAP_PROP_POS_AVI_RATIO)
print cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
print cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
print cap.get(cv2.cv.CV_CAP_PROP_FPS)
print cap.get(cv2.cv.CV_CAP_PROP_FOURCC)
print cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
print cap.get(cv2.cv.CV_CAP_PROP_FORMAT)



ret, frame = cap.read()
