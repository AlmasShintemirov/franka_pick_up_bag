import rclpy
from rclpy.node import Node
# from std_msgs.msg import String
# from std_msgs.msg import Float32MultiArray
# from concurrent.futures import Future
# from rclpy.executors import MultiThreadedExecutor
# from sensor_msgs.msg import Image
# import matplotlib.pyplot as plt
import numpy as np
import time

class Subscriber(Node):
    def __init__(self):
        super().__init__('subscriber')
        self.subscription = None
        self.data = None
        self.future = None

    # def subscribe_once(self, topic_name='/Camera_rgb'):
    #     if self.subscription is None:
    #         if topic_name == '/language_topic':
    #             self.subscription = self.create_subscription(
    #                 String,
    #                 topic_name,
    #                 self.callback,
    #                 10
    #             )
    #         else:
    #             print("subscribing to image topic")
    #             self.subscription = self.create_subscription(
    #                 Image,
    #                 topic_name,
    #                 self.callback,
    #                 10
    #             )
    #         self.future = Future()

    #     return self.future

    def callback(self, msg):
        self.data = msg
        self.get_logger().info('Received data')
        
        self.future.set_result(msg)
        
        self.destroy_subscription(self.subscription)
        self.subscription = None

def wait_for_message(node ,topic_type, topic): 
    class _vfm(object):
        def __init__(self) -> None:
            self.msg = None
            
        def cb(self, msg):
            self.msg = msg

    vfm = _vfm()
    subscription = node.create_subscription(topic_type,topic,vfm.cb,1)
    while rclpy.ok():
        if vfm.msg != None: return vfm.msg
        rclpy.spin_once(node)
        time.sleep(0.001)
    # unsubcription
    subscription.destroy()

def get_observation(node ,topic_type, topic, size=(256,256,3)):

    input_images = wait_for_message(node, topic_type, topic)
    input_images = np.array(input_images.data).reshape(size)

    input_images = np.stack(input_images)[None]
    input_images = np.stack(input_images)[None]

    return input_images
    


# def sub_call(node, topic='/Camera_rgb',size=(512,512,3)):
#     future = node.subscribe_once(topic)
#     print("get the future")

#     executor = MultiThreadedExecutor()
#     while rclpy.ok():
#         rclpy.spin_once(node, timeout_sec=0.1)
        
#         if future.done():
#             break

#         if not size:
#             return future.result()

#         msg_result = future.result()
#         return np.array(msg_result.data).reshape(size)

# class LanguageSubscriber(Node):

#     def __init__(self):
#         super().__init__('language_subscriber')
        
#         self.subscription = self.create_subscription(
#             String,
#             'language_topic',
#             self.listener_callback,
#             10)
#         self.subscription

#     def listener_callback(self, msg):
#         self.get_logger().info('Received: "%s"' % msg.data)
