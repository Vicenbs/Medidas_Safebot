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
        img_analizada = draw_detection_objects(m, net.class_names, objects)

        for obj in objects:
            datos.loc[len(datos)]={"image_name": image_name, "class": net.class_names[int(obj.label)], "confidence": obj.prob*100, "tiempo_inferencia": elapsed}

        cv2.imwrite("resultados/"+image_name+"_results.png", img_analizada)

    datos.to_csv("datos.csv", index=False)
