import cv2
import tensorflow as tf
import numpy as np

def test_drone_video(graph_path='/mnt/data/drone_det/drone_det_rcnn_small.pb',
                     video_path='/mnt/md0/camera_logging/ptz/2019-9-24-18-11-51.mp4'):
    cv2.namedWindow("test")
    gf = tf.GraphDef()
    gf.ParseFromString(open(graph_path, 'rb').read())
    a = [n.name + '=>' + n.op for n in gf.node if n.op in ('Softmax', 'Placeholder')]
    det = [
        "num_detections:0",
        "detection_boxes:0",
        "detection_classes:0",
        "detection_scores:0"
    ]
    tf.import_graph_def(gf, name="")
    sess = tf.Session()
    cap = cv2.VideoCapture(video_path)

    ret = True
    while ret:
        ret, img = cap.read()
        height, width = img.shape[:2]
        d = sess.run(det, {"image_tensor:0": img[np.newaxis]})#/255.})
        for i, (box, label, score) in enumerate(zip(*[t[0] for t in d[1:]])):
            if score < 0.1:
                break
            # if label not in [1.0, 3.0, 2.0, 4.0, 6.0, 8.0, 9.0]:
            #     continue
            save_box2 = int(box[1] * width), int(box[0] * height), int(box[3] * width), int(box[2] * height)
            img = cv2.rectangle(img, (save_box2[0], save_box2[1]), (save_box2[2], save_box2[3]), (0, 255, 0), 1)
            cv2.putText(img, 'c: ' + str(score), (save_box2[0], save_box2[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        cv2.imshow("test", img)
        cv2.waitKey(10)

if __name__ == '__main__':
    test_drone_video()