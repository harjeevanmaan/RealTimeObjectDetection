import tensorflow as tf
import numpy as np
import cv2 
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

def non_max_suppression(inputs, model_size, max_output_size, max_output_size_per_class, iou_threshold, confidence_threshold):
    
    bbox, confs, class_probs = tf.split(inputs, [4, 1, -1], axis=-1)
    bbox = bbox/model_size[0]

    scores = confs * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores = tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=confidence_threshold)
    
    return boxes, scores, classes, valid_detections

def load_class_names(filename):
    with open(filename, 'r') as fp:
        class_names = fp.read().splitlines()
    return class_names

def output_boxes(inputs, model_size, max_output_size, max_output_size_per_class, iou_threshold, confidence_threshold):
    center_x, center_y, width, height, confidence, classes = tf.split(inputs, [1, 1, 1, 1, 1, -1], axis=-1)

    top_left_x = center_x - width/2
    top_left_y = center_y - height/2
    bottom_right_x = center_x + width/2
    bottom_right_y = center_y + height/2

    inputs = tf.concat([top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, classes], axis=-1)
    boxes_dicts = non_max_suppression(inputs, model_size, max_output_size, max_output_size_per_class, iou_threshold, confidence_threshold)

    return boxes_dicts

def draw_outputs(img, boxes, objectness, classes, nums, class_names):
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    boxes = np.array(boxes)

    for i in range(nums):
        x1y1 = tuple((boxes[i, 0:2] * [img.shape[1], img.shape[0]]).astype(np.int32))
        x2y2 = tuple((boxes[i, 2:4] * [img.shape[1], img.shape[0]]).astype(np.int32))

        img = cv2.rectangle(img, (x1y1), (x2y2), (255,0,0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]), (x1y1), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)

    return img

def main():
    model_path = "/home/jeevan/code/tf/yolov3/yolov3.h5"
    class_file = "/home/jeevan/code/tf/yolov3/coco.names"
    imgpath = sys.argv[1]

    model_size = (416, 416, 3)
    num_classes = 80
    max_output_size = 40
    max_output_size_per_class = 20
    iou_threshold = 0.5
    confidence_threshold = 0.5

    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(class_file)

    image = cv2.imread(imgpath)
    image = np.array(image)
    image = tf.expand_dims(image, 0)

    resized_frame = tf.image.resize(image, (model_size[0], model_size[1]))
    prediction = model.predict(resized_frame)

    boxes, scores, classes, nums = output_boxes(
        prediction,
        model_size,
        max_output_size=max_output_size,
        max_output_size_per_class=max_output_size_per_class,
        iou_threshold=iou_threshold,
        confidence_threshold=confidence_threshold
    )

    image = np.squeeze(image)
    img = draw_outputs(image, boxes, scores, classes, nums, class_names)

    win_name = "YoloV3"
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()