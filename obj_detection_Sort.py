#%%
import time
import os
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


# 모델경로 설정

MODEL_PATH ='checkpoints/yolov4-416'

## yolov4-tiny-416 // tiny mode


# 내부 모델 불러오기
saved_model_loaded = tf.saved_model.load( MODEL_PATH , tags = [tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

IOU_THRESHOLD = 0.001


SCORE_THRESHOLD = 0.001
INPUT_SIZE = 416

#%%

def yolo ( img ):
    img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE) , )
    img_input = img_input / 255
    # cv2.imshow('asd', img_input)
    img_input = img_input[np.newaxis, ...].astype(np.float32)

    # yolov4 기반 학습된 모델을 바탕으로 객체 후보군 선정
    img_input = tf.constant(img_input)
    pred_bbox = infer(img_input)

    # non_max_suppression을 활용한 선정된 BOX 종합. 각 임계점에 맞게 탐지된 객체 BOX 선정
    for k, v in pred_bbox.items():
        boxes = v[:, :, 0:4]
        pred_conf = v[:, :, 4:]
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(
            boxes,
            (tf.shape(boxes)[0], -1, 1, 4)
        ),
        scores=tf.reshape(
            pred_conf,
            (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
        ),
        max_output_size_per_class=10000,
        max_total_size=200,
        iou_threshold=IOU_THRESHOLD,
        score_threshold=SCORE_THRESHOLD
    )
    # 선정된 box 저장하기
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    return pred_bbox


#%%
import sort_lib.SORT as SORT
# sort tracker 객체 소환
max_age =  40
min_hits =  1
iou_threshold = 0.01

tracker_test = SORT.Sort( max_age = max_age,
                     min_hits = min_hits,
                     iou_threshold = iou_threshold )



#%%


import cv2
import numpy as np
import time


# 영상 불러오기

for_car = "data/F18003_3_202010210745.avi"

cap = cv2.VideoCapture(for_car)

print(cap.get( cv2.CAP_PROP_FRAME_HEIGHT ))
_ , frame = cap.read()

cnt = 0
boxes = []

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
fps = cap.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)



# 객체 인식 임계점 설정
IOU_THRESHOLD = 0.07  # IOU threshold
SCORE_THRESHOLD = 0.2  # model score threshold

# input size 는 고정
INPUT_SIZE =  416  # resize_img
image_w = width
image_h = height
display = True


# 저장하여 보자
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 코덱 정의
out = cv2.VideoWriter('res.avi', fourcc, fps, (int(image_w), int(image_h)))
print('frame_width : %d, frame_height : %d, fps : %d' %(width, height, fps))

start_time = time.time()

while cap.isOpened():
    ret , frame = cap.read()
    cnt += 1
    if not ret:
        break

    if True:
        # cv2.imshow('asjksadk' , frame )
        #frame = cv2.resize(frame, ( image_w , image_h ))
        tmp_frame = frame.copy()
        dot = yolo( tmp_frame )
        num_class = dot[3][0]
        dot = dot[0][0][:num_class], dot[1][0][:num_class], dot[2][0][:num_class]

        # output 너비 초기화
        for d_ in dot[0]:
            d_[1] = int(d_[1] * image_w)
            d_[3] = int(d_[3] * image_w)
            d_[0] = int(d_[0] * image_h)
            d_[2] = int(d_[2] * image_h)


        ## 사람 가리기
        not_person = dot[2] != 0
        dets = np.hstack([dot[0][not_person], dot[2][not_person].reshape(-1, 1)])
        trackers = tracker_test.update(dets)

        for d in trackers:
            # print(frame, d[4], d[:4])
            d = d.astype(np.int32)

            p1 = d[1], d[0]
            p2 = d[3] , d[2]
            cv2.rectangle( frame , p1 , p2 , ( 3 , 51 , 121 ) , 4 )
            cv2.putText( frame , str(d[4]) , p1 , cv2.FONT_HERSHEY_DUPLEX , 1, (0,0,0) )
        if display :
            cv2.imshow('yolo' , frame )
        out.write(frame)
        print(cnt , end = '\r')


        # time.sleep(10)

    if cv2.waitKey(1) == 27:
        break

print( f"dectection_time : {(time.time() - start_time)/60} min")

out.release()
cap.release()
cv2.destroyAllWindows()




