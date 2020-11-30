#%%
import time
import os
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants


# model path
MODEL_PATH ='checkpoints/yolov4-608'

# import weights
saved_model_loaded = tf.saved_model.load( MODEL_PATH , tags = [tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']




#%%

def yolo ( img ):
    img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE) , )
    img_input = img_input / 255
    img_input = img_input[np.newaxis, ...].astype(np.float32)

    # yolov4 infer
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
    # boxes_dots, class_score, class, ..
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    return pred_bbox


#%%
import sort_lib.SORT as SORT
# sort tracker parameter
max_age =  40
min_hits =  1
iou_threshold = 0.01

# Sort tracker initiation
tracker_test = SORT.Sort( max_age = max_age,
                     min_hits = min_hits,
                     iou_threshold = iou_threshold )



#%%
import cv2
import numpy as np
import time


# load_video
for_car = "data/F18003_3_202010210745.avi"
cap = cv2.VideoCapture(for_car)

# YOLO_parameter
IOU_THRESHOLD = 0.3  # IOU threshold
SCORE_THRESHOLD = 0.5  # model score threshold
INPUT_SIZE =  608 # resize image_size

# initiate_params
cnt = 0
boxes = []
start_time = time.time()

# video_info
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print('frame_width : %d, frame_height : %d, fps : %d' %(width, height, fps))
_ , frame = cap.read()

# resize_img
image_w = width
image_h = height

# display flag
display = True

# generate save_objects
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 코덱 정의
out = cv2.VideoWriter('res.avi', fourcc, fps, (int(image_w), int(image_h)))


while cap.isOpened():
    ret , frame = cap.read()
    cnt += 1
    if not ret:
        break

    if True:
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


        # select vehicle class
        not_person = dot[2] != 0
        dets = np.hstack([dot[0][not_person], dot[2][not_person].reshape(-1, 1)])
        trackers = tracker_test.update(dets)

        # tracking_obj
        for d in trackers:
            d = d.astype(np.int32)
            p1 = d[1], d[0]
            p2 = d[3] , d[2]
            cv2.rectangle( frame , p1 , p2 , ( 3 , 51 , 121 ) , 4 )
            cv2.putText( frame , str(d[4]) , p1 , cv2.FONT_HERSHEY_DUPLEX , 1, (0,0,0) )
        if display :
            cv2.imshow('yolo' , frame )
        out.write(frame)
        print(cnt , end = '\r')

    if cv2.waitKey(1) == 27:
        break

print( f"dectection_time : {(time.time() - start_time)/60} min")

out.release()
cap.release()
cv2.destroyAllWindows()




