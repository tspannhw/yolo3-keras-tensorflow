#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""
import time
import sys
import datetime
import subprocess
import sys
import os
import traceback
import math
import random, string
import base64
import uuid
import json
import psutil
import random, string
from time import gmtime, strftime
import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.3
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
/uid
            else:
keras_yolo3_20180527002205_c43db153-7c2c-4e10-8178-7aab04d6129b.jpg keras_yolo3_20180527002447_e5166f7f-7874-4bcf-8c9d-bccfa9ee6f1c.jpg keras_yolo3_20180527031618_d45226d4-517c-4ead-b260-8b5762c2f9e1.jpg
keras_yolo3_20180527002207_e7958181-ce09-4291-8fb3-e5c4b3ea0a33.jpg keras_yolo3_20180527002450_02e3374d-1466-4055-ad52-3ba30a0b82e2.jpg keras_yolo3_20180527031620_a9611c39-17ab-4aef-955c-a2f14acbf85f.jpg
keras_yolo3_20180527002209_6cec66e5-e769-4c98-954c-1c98712abed3.jpg keras_yolo3_20180527002452_de416eeb-5f46-4336-88fe-2f241569fcab.jpg keras_yolo3_20180527031623_01d434dc-0b2b-461b-bdf2-51c24455cad3.jpg
keras_yolo3_20180527002212_792c4761-856e-4f85-86f9-90bd61280bc4.jpg keras_yolo3_20180527002454_8e1da410-310f-4473-9612-818b8b8ea803.jpg keras_yolo3_20180527031625_a73c2605-67c9-446a-9513-2ed408808ff1.jpg
keras_yolo3_20180527002214_a834c869-1bfb-4fbf-856a-85da074a4df0.jpg keras_yolo3_20180527002456_78de5b38-c1d8-4293-b05e-15ae708be807.jpg keras_yolo3_20180527031627_0715504d-2ce1-4f0a-b044-fdfd51fe7270.jpg
keras_yolo3_20180527002217_5f9f2d08-f20c-4f52-b7cf-28050505847f.jpg keras_yolo3_20180527002459_256458bc-9566-401f-a552-ca68ade330bd.jpg keras_yolo3_20180527031629_d3615ff9-f78f-4ea4-aed1-527daa1e0892.jpg
keras_yolo3_20180527002219_cef00170-094e-45c9-bca8-9dff8526d3a9.jpg keras_yolo3_20180527002501_cb840867-79ba-47fc-bf44-1df5a39dd897.jpg keras_yolo3_20180527031631_f5e244d0-09e4-43e0-8159-f2c84b722626.jpg
keras_yolo3_20180527002221_ac43c713-a3ef-49f7-bff1-49d405b45a84.jpg keras_yolo3_20180527002503_3e590f68-3199-459d-ae7d-cd6e3b4c0647.jpg keras_yolo3_20180527031633_0880f241-4f7a-460e-b114-e0fa49c7ffe1.jpg
keras_yolo3_20180527002224_376889cc-2bff-45c8-9252-b310004dbbf3.jpg keras_yolo3_20180527002506_1f89a89b-3176-4b6c-9917-4670d5aedbd8.jpg keras_yolo3_20180527031636_12f7bf07-2f9e-4a16-a92d-136f56c2c7f6.jpg
keras_yolo3_20180527002226_7e3d4fd5-457f-4c46-9435-53de5af7481f.jpg keras_yolo3_20180527002508_b0faca09-8e29-41a9-b148-fe3924c929f1.jpg keras_yolo3_20180527031638_a37525b5-9653-48a3-8f03-9c287dfac58a.jpg
keras_yolo3_20180527002228_099ef63a-0443-414d-bc24-b9567b86c1c6.jpg keras_yolo3_20180527002511_06d3b470-b1a2-4925-8469-5cc9d533fcb9.jpg keras_yolo3_20180527031640_eb2c371c-db00-40fe-a1e2-b58714390749.jpg
keras_yolo3_20180527002230_6bcd91b0-41c2-4cbb-a920-ef146a331402.jpg keras_yolo3_20180527002513_dc30ea6f-281b-4f28-a456-aee8d2e8568d.jpg keras_yolo3_20180527031642_3e6cab71-8927-4e23-8255-1adda4217480.jpg
keras_yolo3_20180527002233_77e55a0d-101f-4050-a57c-9618dc14dca0.jpg keras_yolo3_20180527002515_b3b66ad0-3ce6-4e57-a2b5-2953dbbb79e0.jpg keras_yolo3_20180527031644_9e176388-9789-468a-8638-7752f916d489.jpg
keras_yolo3_20180527002235_42326786-f982-49c3-b8b5-5ba200a95c57.jpg keras_yolo3_20180527002517_a689e7c9-e49f-45d2-9ee8-22e5f3b219c1.jpg keras_yolo3_20180527031646_3ee41d02-e5f5-4999-8666-0429bf36de8e.jpg
keras_yolo3_20180527002237_7f2cb307-ad7e-4846-a0d7-fd51c8d80be9.jpg keras_yolo3_20180527002519_5fb108dd-f851-46b2-996b-bac2d323efee.jpg keras_yolo3_20180527031648_680a1e9b-b4a8-4627-a32a-818a46316bed.jpg
keras_yolo3_20180527002239_11d9684a-6f3b-4449-889b-e99436c86615.jpg keras_yolo3_20180527002522_5f52a577-a0b6-4de1-a5f7-79b9f7d21e27.jpg keras_yolo3_20180527031651_2321a40b-7d1b-4ca5-8044-3afa0536d8af.jpg
keras_yolo3_20180527002242_a0e5a373-40e3-4a27-b8e8-0155d38395b4.jpg keras_yolo3_20180527002524_a39f90ec-8025-4479-a89e-29d5de265e48.jpg keras_yolo3_20180527031653_e2de26d5-ca31-4b1c-bce8-ce6a510260f4.jpg
keras_yolo3_20180527002244_503aaa1b-cceb-4680-a693-f0509c97d225.jpg keras_yolo3_20180527002526_94f9ec56-7329-494e-809d-ce80d9475949.jpg keras_yolo3_20180527031655_4d4fc92f-e992-47fa-9ee0-66ea8d90a744.jpg
keras_yolo3_20180527002246_8cb1d9b6-0ee2-4da5-8bcd-c68d48f3a8ae.jpg keras_yolo3_20180527002528_2ee61cdc-227f-42ac-8419-372a0c0fdfc5.jpg keras_yolo3_20180527031657_0865ca26-5ac8-476b-a7a9-4e072457c18c.jpg
keras_yolo3_20180527002249_10004c5d-b876-4fc8-8b7c-b9549e4b8ba3.jpg keras_yolo3_20180527002530_64c04119-f224-41db-8879-b51ef8096842.jpg keras_yolo3_20180527031659_a49ff3a7-0427-4f22-8a47-dcf2f519cdc3.jpg
keras_yolo3_20180527002251_fcec7e1b-607c-4097-9557-49e750fae068.jpg keras_yolo3_20180527002532_36efa5e3-6a79-458e-b605-c83c7d93c401.jpg keras_yolo3_20180527031701_3156730c-4e7d-4931-9394-a762df424b87.jpg
keras_yolo3_20180527002254_34596c1e-7da0-446c-9db1-4c3d10c40699.jpg keras_yolo3_20180527002535_b18fd5ca-63d3-4333-adb8-5fd6afba3efd.jpg keras_yolo3_20180527031703_d2fe80a7-392a-495d-964f-7d49544221c0.jpg
keras_yolo3_20180527002257_6691a021-ecca-4d24-a959-e7212312e82b.jpg keras_yolo3_20180527002537_9f1cb2de-1bae-46c6-bb8d-c05255729e68.jpg keras_yolo3_20180527031706_efa2a987-49b2-4654-9c05-bd408cf41a80.jpg
keras_yolo3_20180527002259_8f68e40c-1654-40e7-a545-24287d338956.jpg keras_yolo3_20180527002539_8a477f85-f23f-4979-82e9-a0145c9cb8fe.jpg keras_yolo3_20180527031708_871033c7-4860-4e42-8a89-ff37d30134c6.jpg
keras_yolo3_20180527002302_77dbfa2d-464b-49fe-a1bd-eeab6543a339.jpg keras_yolo3_20180527002541_37adb53e-3ba3-4147-9010-5926f329d7ca.jpg
keras_yolo3_20180527002305_1d7624f7-7d0b-4ea0-8d97-608a561abf2c.jpg keras_yolo3_20180527002543_5e6f8b33-b19a-4ca0-9870-85e9f988e8a3.jpg
➜  images git:(master) ✗ ls -lt keras_yolo3_20180527002025*
zsh: no matches found: keras_yolo3_20180527002025*
➜  images git:(master) ✗ ls -lt keras_yolo3_201805270020*
-rw-r--r--  1 tspann  staff  326514 May 26 20:21 keras_yolo3_20180527002059_69dcaae4-a143-41a3-94a8-f83c677a0314.jpg
-rw-r--r--  1 tspann  staff  344822 May 26 20:20 keras_yolo3_20180527002057_195fd5ea-333f-4bab-9a73-d32333d5b509.jpg
-rw-r--r--  1 tspann  staff  335430 May 26 20:20 keras_yolo3_20180527002055_fa224dd7-5f34-4f4b-a0da-208cc98053c4.jpg
-rw-r--r--  1 tspann  staff  348803 May 26 20:20 keras_yolo3_20180527002053_d695930e-909c-43c1-8e6d-51bceb57ab77.jpg
-rw-r--r--  1 tspann  staff  347261 May 26 20:20 keras_yolo3_20180527002051_863a397c-02a8-4403-b20d-e203c0ecb7ef.jpg
-rw-r--r--  1 tspann  staff  354089 May 26 20:20 keras_yolo3_20180527002049_37bfbe21-8d45-4bcf-aaaa-a12c4d2aa73a.jpg
-rw-r--r--  1 tspann  staff  348207 May 26 20:20 keras_yolo3_20180527002047_1f1bdfa9-3082-42ae-b23c-f15c9833f2ef.jpg
-rw-r--r--  1 tspann  staff  347896 May 26 20:20 keras_yolo3_20180527002044_20cf3ecd-440d-4fcb-b36c-11c5888032d9.jpg
-rw-r--r--  1 tspann  staff  328592 May 26 20:20 keras_yolo3_20180527002042_b1bde11a-10cb-4765-bd2d-0359774c1ec7.jpg
-rw-r--r--  1 tspann  staff  318145 May 26 20:20 keras_yolo3_20180527002040_bcba2593-a2c8-4bf1-86ed-46366dc773d7.jpg
-rw-r--r--  1 tspann  staff  317465 May 26 20:20 keras_yolo3_20180527002038_682b1c01-b361-4378-8ce7-1a52796d7f6e.jpg
-rw-r--r--  1 tspann  staff  312444 May 26 20:20 keras_yolo3_20180527002035_840057f0-db89-4fe1-8082-230bb62ab290.jpg
-rw-r--r--  1 tspann  staff  318081 May 26 20:20 keras_yolo3_20180527002033_a81242cd-407d-4175-9f83-593c9de96c67.jpg
-rw-r--r--  1 tspann  staff  313191 May 26 20:20 keras_yolo3_20180527002031_88bfcd04-76d9-4227-b910-86b316868613.jpg
-rw-r--r--  1 tspann  staff  306105 May 26 20:20 keras_yolo3_20180527002029_1935215b-098f-469d-bf8a-e21fefd899d9.jpg
-rw-r--r--  1 tspann  staff  337092 May 26 20:20 keras_yolo3_20180527002026_4d718065-3ef9-401c-b9d7-a96355b1076f.jpg
➜  images git:(master) ✗ cd ..
➜  keras-yolo3 git:(master) ✗ vinifi.py
➜  keras-yolo3 git:(master) ✗ vi yolonifi.py
^[[A
➜  keras-yolo3 git:(master) ✗ vi yolonifi.py
➜  keras-yolo3 git:(master) ✗ vi yolonifi.py
➜  keras-yolo3 git:(master) ✗ cat yolonifi.py
#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a YOLO_v3 style detection model on test images.
"""
import time
import sys
import datetime
import subprocess
import sys
import os
import traceback
import math
import random, string
import base64
import uuid
import json
import psutil
import random, string
from time import gmtime, strftime
import colorsys
import os
import random
from timeit import time
from timeit import default_timer as timer  ### to calculate FPS

import numpy as np
from keras import backend as K
from keras.models import load_model
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval
from yolo3.utils import letterbox_image

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.score = 0.3
        self.iou = 0.5
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.model_image_size = (416, 416) # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model must be a .h5 file.'

        self.yolo_model = load_model(model_path, compile=False)
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        global yoloid
        start = time.time()

        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        row = { }
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        row['boxes'] = 'Found {} boxes for {}'.format(len(out_boxes), 'img')

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            row['class{0}'.format(i)] = '{0}'.format(str(predicted_class))
            row['score{0}'.format(i)] = '{0}'.format(str(score))
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            row['left{0}'.format(i)] = '{0}'.format(str(left))
            row['right{0}'.format(i)] = '{0}'.format(str(right))
            row['top{0}'.format(i)] = '{0}'.format(str(top))
            row['bottom{0}'.format(i)] = '{0}'.format(str(bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = time.time()
        print(end - start)
        row['host'] = os.uname()[1]
        row['end'] = '{0}'.format( str(end ))
        row['te'] = '{0}'.format(str(end-start))
        row['battery'] = psutil.sensors_battery()[0]
        row['systemtime'] = datetime.datetime.now().strftime('%m/%d/%Y %H:%M:%S')
        row['cpu'] = psutil.cpu_percent(interval=1)
        usage = psutil.disk_usage("/")
        row['diskusage'] = "{:.1f} MB".format(float(usage.free) / 1024 / 1024)
        row['memory'] = psutil.virtual_memory().percent
        yoloid = '{0}_{1}'.format(strftime("%Y%m%d%H%M%S",gmtime()),uuid.uuid4())
        row['yoloid'] = str(yoloid)
        json_string = json.dumps(row)
        filename = 'keras_yolo3_{0}.json'.format(yoloid)
        fh = open("/Volumes/seagate/projects/keras-yolo3/logs/" + filename, "a")
        fh.writelines('{0}\n'.format(json_string))
        fh.close

        return image

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path):
    global yoloid
    import cv2
    vid = cv2.VideoCapture(1) # video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        imgfilename = '/Volumes/seagate/projects/keras-yolo3/images/keras_yolo3_{0}.jpg'.format(yoloid)
        cv2.imwrite(imgfilename, result)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()



if __name__ == '__main__':
    detect_img(YOLO())
