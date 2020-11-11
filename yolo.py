import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
#from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# deep sort imports
#from deep_sort import preprocessing
#from deep_sort.detection import Detection
#from deep_sort.tracker import Tracker
#from tools import generate_detections as gdet
import pickle

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('out_video', None, 'path to output video')
flags.DEFINE_string('out_xml', None, 'path to output xml')
flags.DEFINE_float('iou', 0.1, 'iou threshold')
flags.DEFINE_float('score', 0.01, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')

def main(_argv):
    s_t = time.time()
    
    # initialize tracker
    nms_max_overlap = 0.95
    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set

        
    # otherwise load standard tensorflow saved model

    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
    infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.out_video:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(FLAGS.out_video, codec, fps, (width, height))
        
    frame_cnt = 0
    car_info = []
    # set colors for classes
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    color = {"car":colors[0],"bus":colors[1],'motorbike':colors[2],'truck':colors[3]}
    
    # find video name
    for ii in range(len(FLAGS.video)):
            if FLAGS.video[ii] == "F":
                break
    v = FLAGS.video[ii:-4]
    
    # while video is running
    while True:
        return_value, frame = vid.read()

        
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        
        
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections
        batch_data = tf.constant(image_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]
        
        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car','bus','motorbike','truck']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            w = bboxes[i][2]-bboxes[i][0]
            h = bboxes[i][3]-bboxes[i][1]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            elif (w>original_w/3)or(h>original_h/3):
                deleted_indx.append(i)
            else:
                names.append(class_name)
        
        names = np.array(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        indices = preprocessing.non_max_suppression(bboxes, classes, nms_max_overlap, scores)
        # update tracks
        for ind in indices:
            bbox = bboxes[ind]
            class_name = names[ind]
            x,y,w,h = bbox
            car_info.append({"frame_no":frame_cnt,"class":class_name,"x":x,"y":y,"w":w,"h":h})

            # draw bbox on screen
            c = color[class_name]
            c = [i * 255 for i in c]
            cv2.rectangle(frame, (int(x), int(y)), (int(x)+int(w), int(y)+int(h)), c, 2)
            cv2.putText(frame, class_name + "-" + str(ind),(int(bbox[0]), int(bbox[1])+10),0, 0.5, (0,0,0),2)
            
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print(f"{v+str(frame_cnt)} : {fps}")

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.out_video:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            with open(f'{v}.pickle','wb') as fw:
                pickle.dump(car_info, fw)
            break
            
    frame_cnt+=1
    
    print(time.time()-s_t)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass