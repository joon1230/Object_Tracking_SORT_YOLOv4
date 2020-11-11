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
with open( "data/yolo_dectetion_dong.pickle" , 'rb' ) as f:
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

#%%

"""
매 프레임 별로 탐지된 데이터 == dets
$$dets$$
 shape : ( 탐지된 객체수, 4 ) 
 tpye : list or np.array 

를 탐지한 프레임 마다 tracker 객체에게 dets 를 부여하면 
trackers의에 갱신된 정보가 저장됩니다. 

trackers에는 가장 최근에 갱신된 tracker 번호와, 그 해당객체의 좌표가 저장되어 집니다.
trackers 안의 개별 객체 d는 [ x1 , y1 , x2, y2 , tracker_id ] 로 저장되어 있습니다~

아래의 코드는 객체 식별은 2프레임당 1개씩 했으므로 영상과 비교하기위해 2프레임당 1개씩 출력하고 거기에 따른
tracking 결과가 보여집니다~ 

"""

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
        try:
            """
            입력 데이터 입니다~
            """
            # dets = np.hstack([boxes[f][0], boxes[f][1].reshape(-1, 1)])
            dets = boxes[f]
        except:
            pass
        """
        tracker 갱신 구간 입니다.
        """
        trackers = tracker_test.update(dets)


        """
        갱신된 tracker 들을 보여주는 부분입니다~~
        """
        for d in trackers:
            print(frame, d[4], d[:4])
            d = d.astype(np.int32)
            p1 = d[0], d[1]
            p2 = d[2] , d[3]
            cv2.rectangle( frame , p1 , p2 , ( 3 , 51 , 121 ) , 2 )
            cv2.putText( frame, str(d[4]) , p1 , cv2.FONT_HERSHEY_DUPLEX , 1, (0,0,0) )

        cv2.imshow('tracking', frame)
        out.write(frame)

        # time.sleep(10)

    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()
cap.release()

print( time.time() - start )


#%%
""" 
영상출력 없이 갱신된 tracker의 list 만 출력합니다~
"""

total_frames = 0

start = time.time()
for frame in range(len(boxes)):
    # dets = np.hstack([boxes[frame][0], boxes[frame][1].reshape(-1,1)])
    dets = boxes[frame]
    frame += 1
    total_frames += 1
    trackers = tracker_test.update(dets)

    for d in trackers:
        print( f"frame : {frame} , tracker_id : {int(d[4])} ,  points : {d[:4]}")
print( time.time() - start )
