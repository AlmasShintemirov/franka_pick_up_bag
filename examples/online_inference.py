import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from concurrent.futures import Future
from rclpy.executors import MultiThreadedExecutor
import matplotlib.pyplot as plt
import numpy as np

class MultiTopicSubscriber(Node):
    def __init__(self):
        super().__init__('multi_topic_subscriber')
        # 初始化订阅者和Future对象
        self.language_subscription = None
        self.wrist_image_subscription = None
        self.camera_image_subscription = None

        self.language_data = None
        self.wrist_image_data = None
        self.camera_image_data = None

        self.language_future = None
        self.wrist_image_future = None
        self.camera_image_future = None

        self.subscription_count = 0  # 用于计数接收到的消息数

    # 订阅 language_topic
    def subscribe_language_once(self, topic_name='/language_topic'):
        if self.language_subscription is None:
            self.language_subscription = self.create_subscription(
                String,
                topic_name,
                self.language_callback,
                10
            )
            self.language_future = Future()

        return self.language_future

    # 订阅 Camera_wrist_image
    def subscribe_wrist_image_once(self, topic_name='/Camera_wrist_rgb'):
        if self.wrist_image_subscription is None:
            self.wrist_image_subscription = self.create_subscription(
                Float32MultiArray,
                topic_name,
                self.wrist_image_callback,
                10
            )
            self.wrist_image_future = Future()

        return self.wrist_image_future

    # 订阅 Camera_image
    def subscribe_camera_image_once(self, topic_name='/Camera_rgb'):
        if self.camera_image_subscription is None:
            self.camera_image_subscription = self.create_subscription(
                Float32MultiArray,
                topic_name,
                self.camera_image_callback,
                10
            )
            self.camera_image_future = Future()

        return self.camera_image_future

    # 各个话题的回调函数
    def language_callback(self, msg):
        self.language_data = msg
        self.get_logger().info(f'Received language message: {msg.data}')
        self.language_future.set_result(msg)
        self.destroy_subscription(self.language_subscription)
        self.language_subscription = None

    def wrist_image_callback(self, msg):
        self.wrist_image_data = msg
        self.get_logger().info('Received wrist camera image')
        self.wrist_image_future.set_result(msg)
        self.destroy_subscription(self.wrist_image_subscription)
        self.wrist_image_subscription = None

    def camera_image_callback(self, msg):
        self.camera_image_data = msg
        self.get_logger().info('Received camera image')
        self.camera_image_future.set_result(msg)
        self.destroy_subscription(self.camera_image_subscription)
        self.camera_image_subscription = None

def main(args=None):
    rclpy.init(args=args)
    node = MultiTopicSubscriber()

    # 同时订阅多个不同类型的消息
    language_future = node.subscribe_language_once()
    wrist_image_future = node.subscribe_wrist_image_once()
    camera_image_future = node.subscribe_camera_image_once()

    # 使用executor来运行节点并等待结果
    executor = MultiThreadedExecutor()    

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)

        if language_future.done() and wrist_image_future.done() and camera_image_future.done():
            break

    # 获取到的消息
    if language_future.done():
        language_msg = language_future.result()
        node.get_logger().info(f'Final Language Message: {language_msg.data}')

    if wrist_image_future.done():
        wrist_image_msg = wrist_image_future.result()
        node.get_logger().info(f'Final Wrist Image Data')
        camera_image_array = np.array(wrist_image_msg.data).reshape((512, 512, 3))
        plt.imshow(camera_image_array)

    if camera_image_future.done():
        camera_image_msg = camera_image_future.result()
        node.get_logger().info(f'Final Camera Image Data: {camera_image_msg.data}')
        camera_image_array = np.array(camera_image_msg.data).reshape((512, 512, 3))
        plt.imshow(camera_image_array)


    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
