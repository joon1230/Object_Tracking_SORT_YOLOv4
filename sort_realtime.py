#%%

import pickle
import numpy as np
import os
### 현재 실행되는 python 파일의 위치 입력
os.chdir( "C:/Users/user/workdiratory/SORT" )

import sort_lib.SORT as SORT
import cv2

"""
tracker 객체의 parameter 정의 구간입니다.
"""
max_age =  20
min_hits =  10
iou_threshold = 0.3

tracker_test = SORT.Sort( max_age = max_age,
                     min_hits = min_hits,
                     iou_threshold = iou_threshold )


# load_video
for_car = "data/F18003_3_202010210745.avi"

# video 정보
cap = cv2.VideoCapture(for_car)
print(cap.get( cv2.CAP_PROP_FRAME_HEIGHT ))
_ , frame = cap.read()
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
fps = cap.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)

# save 객체
fourcc = cv2.VideoWriter_fourcc(*'DIVX') # 코덱 정의
out = cv2.VideoWriter('res.avi', fourcc, fps, (int(width), int(height)))

"""
data shape 정의
매 프레임 마다 객체 식별후 tracker 에게
"""

start_time = time.time()
cnt = 0
while cap.isOpened():
    ret , frame = cap.read()
    cnt += 1
    if not ret:
        break

    """
    프레임당 객체를 식별 하고
    """
    if cnt % 6 != 0:
        # cv2.imshow('asjksadk' , frame )
        tmp_frame = frame.copy()
        _ ,_ , dot = yolo( tmp_frame )
        num_class = dot[3][0]
        dot = dot[0][0][:num_class], dot[1][0][:num_class], dot[2][0][:num_class]
        dets = np.hstack([dot[0], dot[1].reshape(-1, 1)])

        """
        프레임의 탐지된 객체들을 tracker에 부여 및 tracker 갱신합니다
        $$dets$$
        shape : ( 해당프레임 탐지된 객체수 , 4 )
        구조 : [ 
                [ x1, y1, x2, y2 ],
                [ x1, y1, x2, y2 ],..
            ]
        """
        trackers = tracker_test.update(dets)

        """
        프레임별 갱신된 정보를 출력합니다
        """
        for d in trackers:
            # print(frame, d[4], d[:4])
            d = d.astype(np.int32)
            p1 = d[1], d[0]
            p2 = d[3] , d[2]
            cv2.rectangle( frame , p1 , p2 , ( 14 , 255 , 0 ) , 1 )
            cv2.putText( frame , str(d[4]) , p1 , cv2.FONT_HERSHEY_DUPLEX , 1, (0,0,0) )

        cv2.imshow('yolo' , frame )
        out.write(frame)

        # time.sleep(10)

    if cv2.waitKey(1) == 27:
        break

print( f"총 소요 시간 : {(time.time() - start_time)/60} 분")

out.release()
cap.release()
cv2.destroyAllWindows()
 # ram할당 제거