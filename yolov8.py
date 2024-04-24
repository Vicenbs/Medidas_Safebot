# Tencent is pleased to support the open source community by making ncnn available.
#
# Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

import sys
import cv2
from ncnn.model_zoo import get_model
from ncnn.utils import draw_detection_objects
import os
import pandas as pd
import time

if __name__ == "__main__":
    image_folder = "images"
    image_files = os.listdir(image_folder)
    net = get_model(
            "yolov8s",
            target_size=640,
            prob_threshold=0.25,
            nms_threshold=0.45,
            num_threads=4,
            use_gpu=True,
        )
    datos = pd.DataFrame(columns=["image_name", "class", "confidence", "tiempo_inferencia"])

    for image_name in image_files:
        imagepath = os.path.join(image_folder, image_name)
        m = cv2.imread(imagepath)
        if m is None:
            print("cv2.imread %s failed\n" % (imagepath))
            sys.exit(0)
        t = time.time()
        objects = net(m)
        elapsed = time.time() - t
        draw_detection_objects(m, net.class_names, objects)

        for obj in objects:
            datos = datos.append({"image_name": image_name, "class": net.class_names[int(obj.label)], "confidence": obj.prob*100, "tiempo_inferencia": elapsed}, ignore_index=True)

        cv2.imwrite("resultados/"+image_name+"_results.png", m)

    datos.to_csv("datos.csv", index=False)
