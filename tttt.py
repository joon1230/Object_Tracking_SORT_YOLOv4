
#%%

import pickle
import numpy as np
import os

### 현재 실행되는 python 파일의 위치 입력
os.chdir( "C:/Users/user/workdiratory/SORT" )
import sort_lib.SORT as SORT
import cv2




#%%

# 비교를 위한 입력 비디오
VIDEO = "data/F18003_3_202010210745.avi"
cap = cv2.VideoCapture(VIDEO)

print(cap.get( cv2.CAP_PROP_FRAME_HEIGHT ))


## video 저장을 위한 기본 값 선언
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('res.avi', fourcc, fps, (int(width), int(height)))


#%%
""" 
매 프레임 별로 탐지된 객체의 정보가 담겨야 합니다.
입력 데이터 
$$boxes$$ 
 shape은 : ( 프레임수 , 해당프레임에 탐지된 객체의 수 , 4 )
 tpye : list or np.array
 ex ) boxes[0] 은 0번째 프레임에 탐지된 모든 객체의 박스좌표 ( x1, y1, x2, y2 )
"""



# 프레임별 탐지된 모든 객체 박스 정보 불러오기
with open( "data/yolo_dectetion_num_class,.pickle" , 'rb' ) as f:
    boxes = pickle.load(f )

# sort tracker 객체 호출

"""
tracker 객체의 parameter 정의 구간입니다.
"""
max_age =  20
min_hits =  10
iou_threshold = 0.3

tracker_test = SORT.Sort( max_age = max_age,
                     min_hits = min_hits,
                     iou_threshold = iou_threshold )





class_dict = {0: 'car', 1: 'truck', 2: 'bus'}

_ , frame = cap.read()
import time
cnt = 0
f = 0
start = time.time()
while cap.isOpened():
    ret , frame = cap.read()
    cnt += 1
    if not ret:
        break

    if cnt < 5:
        pass
    elif cnt % 2 != 0: # interval_frame 프레임 간격
        f += 1

        dets = boxes[f]

        for i,d in enumerate(dets):
            print(i)
            print(frame, d[4], d[:4])
            d = d.astype(np.int32)
            p1 = d[0], d[1]
            p2 = d[2] , d[3]
            cv2.rectangle( frame , p1 , p2 , ( 3 , 51 , 121 ) , 2 )
            cv2.putText( frame, class_dict[d[-1]] + str(d[4]) , p1 , cv2.FONT_HERSHEY_DUPLEX , 0.7, (0,0,0) )

        cv2.imshow('tracking', frame)
        out.write(frame)

        # time.sleep(10)

    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()
cap.release()

print( time.time() - start )