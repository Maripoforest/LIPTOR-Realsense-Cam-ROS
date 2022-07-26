import rospy
from sensor_msgs.msg import Image as msg_Image
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
import os
import pyrealsense2 as rs2
import numpy as np
from cvimage_msgs.msg import CvImage

knt = 0
kntc = 0

class ImageListener:
    def __init__(self, topic):
        self.topic = topic
        self.bridge = CvBridge()
        self.vector = Vector3()
        self.cvImage_color = CvImage()
        self.cvImage_depth = CvImage()
        self.sub = rospy.Subscriber(topic, msg_Image, self.imageDepthCallback)
        self.sub_info = rospy.Subscriber('/camera/depth/camera_info', CameraInfo, self.imageDepthInfoCallback)
        self.sub_color = rospy.Subscriber('/camera/color/image_raw', msg_Image, self.imageColorCallback)
        self.pub_color = rospy.Publisher('/camera_image_color', CvImage, queue_size=10)
        self.pub_depth = rospy.Publisher('/camera_image_depth', CvImage, queue_size=10)
        self.pub_coord = rospy.Publisher('/object_coordination',Vector3, queue_size=10)
        self.intrinsics = None

    def imageDepthCallback(self, data):
        global knt
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)

            w = int(data.width/2)
            h = int(data.height/2)
   
            # sys.stdout.write('%s: Depth at center(%d, %d): %f(mm)\r' % (self.topic, pix[0], pix[1], cv_image[pix[1], pix[0]]))
            # sys.stdout.flush()
            if self.intrinsics:
                while (cv_image[w, h] == 0):
                    w += 1
                    if (w - int(data.width/2) >= 10):
                        w = int(data.width/2)
                        break
                while (cv_image[w, h] == 0):
                    h += 1
                    if (h - int(data.height/2) >= 6):
                        h = int(data.height/2)
                        break
                while (cv_image[w, h] == 0):
                    w -= 1
                    if (w - int(data.width/2) <= -10):
                        w = int(data.width/2)
                        break
                while (cv_image[w, h] == 0):
                    h -= 1
                    if (h - int(data.height/2) <= -6):
                        h = int(data.height/2)
                        break
                print(w, h)

                depth = cv_image[w, h]
                result = rs2.rs2_deproject_pixel_to_point(self.intrinsics, [w, h], depth)
                print(result)
                self.vector.x = result[2]
                self.vector.y = result[0]
                self.vector.z = 0 - result[1]
                self.pub_coord.publish(self.vector)

                knt += 1
                knt %= 2

                if knt == 0:

                    cv_image = cv2.resize(cv_image, (200, 150))
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    cv_image = cv2.applyColorMap(cv2.convertScaleAbs(cv_image, alpha=0.03), cv2.COLORMAP_JET)
                    size = cv_image.shape[0] * cv_image.shape[1] * cv_image.shape[2]
                    data = list(cv_image.reshape(size))
                    # cv2.imshow("Depth", cv_image)
                    # cv2.waitKey(1)
                    
                    self.cvImage_depth.data = data
                    self.cvImage_depth.size = list(cv_image.shape)
                    self.cvImage_depth.time = rospy.get_time()
            
                    self.pub_depth.publish(self.cvImage_depth)
                    # sys.stdout.write('%s: Depth at center(%d, %d): %f(mm)\r' % (self.topic, pix[0], pix[1], cv_image[320, 240]))
                    # sys.stdout.flush()

        except CvBridgeError as e:
            print(e)
            return

    def imageDepthInfoCallback(self, cameraInfo):
        try:
            # import pdb; pdb.set_trace()
            if self.intrinsics:
                return
            self.intrinsics = rs2.intrinsics()
            self.intrinsics.width = cameraInfo.width
            self.intrinsics.height = cameraInfo.height
            self.intrinsics.ppx = cameraInfo.K[2]
            self.intrinsics.ppy = cameraInfo.K[5]
            self.intrinsics.fx = cameraInfo.K[0]
            self.intrinsics.fy = cameraInfo.K[4]
            if cameraInfo.distortion_model == 'plumb_bob':
                self.intrinsics.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.intrinsics.model = rs2.distortion.kannala_brandt4
            self.intrinsics.coeffs = [i for i in cameraInfo.D]
        except CvBridgeError as e:
            print(e)
            return
    
    def imageColorCallback(self, data):
        global kntc
        
        cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')   
        cv2.imshow("color", cv_image)
        cv2.waitKey(1)

        kntc += 1
        kntc %= 2

        if kntc == 0:

            cv_image = cv2.resize(cv_image, (200, 150))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            size = cv_image.shape[0] * cv_image.shape[1] * cv_image.shape[2]
            data = list(cv_image.reshape(size))
            
            self.cvImage_color.data = data
            self.cvImage_color.size = list(cv_image.shape)
            self.cvImage_color.time = rospy.get_time()
    
            self.pub_color.publish(self.cvImage_color)
        # if (key == 27):
        #     rospy.signal_shutdown("Keyboard Interruption")

    

def main():
    topic = '/camera/depth/image_rect_raw'
    listener = ImageListener(topic)
    rospy.spin()

if __name__ == '__main__':
    node_name = os.path.basename(sys.argv[0]).split('.')[0]
    rospy.init_node(node_name)
    main()