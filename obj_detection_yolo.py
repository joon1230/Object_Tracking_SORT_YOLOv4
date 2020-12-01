#%%

import time
import os
os.chdir(os.path.dirname(__file__))
from shapely.geometry import LineString


#%%

# assign GPU
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
except:
    print('NO GPU')


import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

# model_load
MODEL_PATH ='checkpoints/yolov4-608' # yolov4-608, yolov4-416
saved_model_loaded = tf.saved_model.load( MODEL_PATH , tags = [tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']


#%%

def yolo ( img ):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
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
    # 선정된 box 저장하기
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    return pred_bbox


def reverse_points( p ):
    y_min = p[0]
    x_min = p[1]
    y_max = p[2]
    x_max = p[3]
    return [ y_max, x_min, y_min, x_max]


# roi filter
def borderline(pts):
    border = []
    for i in range(len(pts)-1):
        border.append(LineString([pts[i], pts[i+1]]))
    border.append(LineString([pts[-1], pts[0]]))
    return border

def point_in_border(point, pts):
    border = borderline(pts)
    com_line = LineString([(0, point[0]), (point[1], point[0])])
    count = 0

    for line in border:
        if com_line.intersection(line):
            count += 1
    if count%2 == 0:
        return False
    else:
        return True



#%%
import cv2
import numpy as np
import time


# yolo hyper parameter
IOU_THRESHOLD = 0.3  # IOU threshold
SCORE_THRESHOLD = 0.3  # model score threshold
INPUT_SIZE =  608  # resize_img


# CONSTANCE
SAMPLE_FRAME = 2000
RESIZE_RATIO = 1
DISPLAY = False
with open( 'checkpoints/coco.names' , 'r' ) as f:
    lab = f.read().split('\n')
cls_dict = dict( zip([int(i) for i in range(len(lab))] , lab))
cls_xml = { 2:'승용/승합', 5:"버스", 7:"화물/기타", 3:"이륜차",  6:"버스" , 1 : "이륜차" }
cls_color = { 2: (255,204,92) , 5 : (255,111,105) , 7 : (150,206,180) ,
              3 : (66,139,202) , 6 : (255,111,105), 1 : (66,139,202) }

# angle roi
pts = np.array([[295,25],
 [295,613],
 [1043,630],
 [1016,442],
 [1061,342],
 [1022,61]] , dtype = np.int32)

# path info
save_path = "output"
video_path = "data/test.avi"

# init params
cnt = 0
save_cnt = 0

# video load & info
cap = cv2.VideoCapture(video_path)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print('frame_width : %d, frame_height : %d, fps : %d, total_frame : %d' % (width, height, fps, length ))

# resize img & flag
image_w = int(width*RESIZE_RATIO)
image_h = int(height*RESIZE_RATIO)
display = DISPLAY

# generate save_objects
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter( os.path.join( save_path, "res_yolo.avi") , fourcc, int(fps), (int(image_w), int(image_h)))

# check time
start_time = time.time()



while cap.isOpened():
    ret , frame = cap.read()
    cnt += 1

    if not ret:
        break

    if cnt < SAMPLE_FRAME:
        frame = cv2.resize(frame, ( image_w , image_h ))
        tmp_frame = frame.copy()
        dot = yolo( tmp_frame ) # detection by yolo
        num_class = dot[3][0]
        dot = dot[0][0][:num_class], dot[1][0][:num_class], dot[2][0][:num_class]

        # output 너비 초기화
        for d_ in dot[0]:
            d_[1] = int(d_[1] * image_w) # x_min
            d_[3] = int(d_[3] * image_w) # x_max
            d_[0] = int(d_[0] * image_h) # y_min
            d_[2] = int(d_[2] * image_h) # y_max

        # select vechicle
        select_obj = np.isin( dot[2] , [1,2,3,5,7])

        # filter boxes
        dets = np.hstack([dot[0], dot[1].reshape(-1, 1)])
        obj = np.hstack([dot[0][select_obj], dot[2][select_obj].reshape(-1, 1)]).astype(np.int32)
        p2_in = [ point_in_border( o[2:] , pts ) for o in obj]
        p1_in = [ point_in_border( o[0:2] , pts ) for o in obj]
        if len(p2_in) * len(p1_in) == 0:
            is_in = []
        else:
            try:
                is_in = np.array(p1_in) | np.array(p2_in)
            except:
                print("\n\n")
                print("ERROR!!!!")
                print(p2_in, p1_in)
                break

        for d in obj[is_in]:
            d = d.astype(np.int32)

            p1 = d[1], d[0]
            p2 = d[3] , d[2]

            cv2.rectangle( frame , p1 , p2 , cls_color.get(int(d[4]), (255,204,92) ) , 3 )
            cv2.putText( frame , cls_dict[int(d[4])] , p1 , cv2.FONT_HERSHEY_DUPLEX , 1, (0,0,0) )

        cv2.polylines(frame, [pts] , True , (0,0,255) , 4)
        cv2.polylines(tmp_frame, [pts], True, (0, 0, 255), 4)
        if display :
            cv2.imshow('yolo' , frame )

        # save
        out.write(frame)
        save_cnt += 1
        print(f"progress : {int((cnt / length) * 100)}%",end = '\r')
    else:
        break


print( f"{(time.time() - start_time) / 60} min" )
print( "\n\n\n" )
out.release()
cap.release()
