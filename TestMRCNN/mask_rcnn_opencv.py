import cv2
import pyrealsense2 as rs
import numpy as np
# from mrcnn import visualize

# RealSense Initialize
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
point_x, point_y = 250, 100

# DNN
net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                    "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")


while True:

    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    # print(depth_image)
    distance_mm = depth_image[point_y, point_x]
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    cv2.circle(color_image, (point_x, point_y), 8, (0, 0, 255), -1)
    cv2.putText(color_image, "{} mm".format(distance_mm), (point_x, point_y - 10), 0, 1, (0, 0, 255), 2)

    blob = cv2.dnn.blobFromImage(color_image, swapRB=True)
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    detection_count = boxes.shape[2]
    print(detection_count)

    # Boxes and masks
    for i in range(detection_count):
        box = boxes[0, 0, i]
        class_id = box[1]
        score = box[2]
        height, width, _ = color_image.shape

        x = int(box[3] * width)
        y = int(box[4] * height)
        x2 = int(box[5] * width)
        y2 = int(box[6] * height)
        # roi = color_image[y: y2, x: x2]
        # roi_height, roi_width, _ = roi.shape

        # mask = masks[i, int(class_id)]
        # mask = cv2.resize(mask, (roi_width, roi_height))
        # _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)

        # colors = visualize.random_colors(80)
        # contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # for cnt in contours:
            # cv2.polylines(color_image, [cnt], True, colors[i], 2)
            # img = visualize.draw_mask(color_image, [cnt], colors[i])

        cv2.rectangle(color_image, (x, y), (x2, y2), (255, 0, 0), 2)
        cv2.imshow("Img", color_image)
        # cv2.imshow("Mask", mask)
        key = cv2.waitKey(1)
        if (key == 27):
            pipeline.stop()
            break

    # cv2.imshow("Image", img)
    # cv2.imshow("2", depth_colormap)

    # key = cv2.waitKey(1)
    # if (key == 27):
    #     pipeline.stop()
    #     break