

## sort study!

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
numpy

-- weights
//yolov4-416
//yolov4-608

requirements.txt 참고
```

### Demo:
To run the tracker with the provided detections:
```
sort_oneshot.py 식별된 영상 tracking
obj_detection_Sort.py  realtime detecting & tracking
```

sort_oneshot.py : 탐지된 박스 정보를 가지고 객체 추적

obj_detection_Sort.py : 매 프레임 별로 탐지( yolov4 기반 ) , 추적( SORT )
