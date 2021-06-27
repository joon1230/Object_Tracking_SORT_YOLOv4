

## 객체 탐지 및 추적 ( Sort + YOLO )
객체 탐지 모델인 yolov4와 객체 추적 알고리즘인 SORT를 활용하여 실시간 객체 탐지 및 추적 project 에 사용한 방법론 입니다.
도로변의 차량을 식별및 추적을 위해 구현된 코드 입니다.

----- 
origin - https://github.com/abewley/sort

reference - https://arxiv.org/abs/1602.00763


-----
#### package 요구사항 :

```
-- tracking
filterpy==1.4.5
lap==0.4.0 ## ignore.
scipy

-- dectection
tensorflow==2.3
shapely
numpy

-- weights ( for yolov4 )
//yolov4-416
//yolov4-608

requirements.txt 참고
```


### Demo:
To run the tracker with the provided detections:
```
$ python sort_oneshot.py
```

To run object detection YOLOv4( 608 ) :
```
$ python obj_detection_yolo.py
```

To run Yolov4 + tracker
```
$ python YOLO_Sort.py  // realtime detecting & tracking ( demo )
```

