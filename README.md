

## sort study!

origin - https://github.com/abewley/sort

reference - https://arxiv.org/abs/1602.00763

-----
#### package 요구사항 :

```
filterpy==1.4.5
lap==0.4.0 ## 무시가능
scipy

requirements.txt 참고
```

### Demo:
To run the tracker with the provided detections:
```
sort_oneshot.py 식별된 영상 tracking
sort_realtime.py  realtime detecting & tracking
```

sort_oneshot.py : 영상내에서 매 프레임별로 탐지된 모든 객체의 정보가 저장된 파일을 기반으로 tracking 합니다.
따라서 모든프레임 별로 객체 박스정보가 담긴 list 형태의 파일이 필요합니다.

sort_realtime.py : 영상을 받아 매프레임 별로 detecting과 tracking을 실시합니다. detection 부분이 필요합니다 추가 보강중..
