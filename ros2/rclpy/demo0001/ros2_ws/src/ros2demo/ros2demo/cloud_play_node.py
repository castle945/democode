#!/usr/bin/env python
# ROS2 节点写法，参考 https://github.com/HaiderAbasi/ROS2-Path-Planning-and-Maze-Solving/blob/master/path_planning_ws/src/maze_bot/maze_bot/maze_solver.py https://github.com/ros2/examples/blob/rolling/rclpy/topics/pointcloud_publisher/examples_rclpy_pointcloud_publisher/pointcloud_publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ros2demo.rospy_msgs.cloud_publisher import CloudPublisher
from glob import glob
import numpy as np

class CloudPlayNode(Node):
    def __init__(self):
        super().__init__('cloud_play_node')
        self.key_sub = self.create_subscription(String, '/teleop_key', self.key_callback, 10)
        self.cloud_pub = CloudPublisher(node=self, topic_name="/lidar", frame_id="ego", qos_profile=1, point_type="PointXYZI")
        self.timer = self.create_timer(1.0, self.process) # 主逻辑处理函数，spin 节点时执行。话题订阅的回调函数执行频率与 spin 无关，只与发布频率有关，有消息到就执行

        self.declare_parameter('data_root', '/datasets/KITTI/object/training/velodyne')
        self.data_root = self.get_parameter('data_root').get_parameter_value().string_value

        self.filepaths = sorted(glob(f'{self.data_root}/*'))
        self.idx, self.len, self.play = 0, len(self.filepaths), False

    def key_callback(self, data, step=10):
        keycode = data.data
        if keycode in ['w', 'W']:
            print(f"keycode w, idx - {step}")
            self.idx = (self.idx - step + self.len) % self.len
        elif keycode in ['s', 'S']:
            print(f"keycode s, idx + {step}")
            self.idx = (self.idx + step + self.len) % self.len
        elif keycode in ['a', 'A']:
            print(f"keycode a, idx - 1")
            self.idx = (self.idx - 1 + self.len) % self.len
        elif keycode in ['d', 'D']:
            print(f"keycode d, idx + 1")
            self.idx = (self.idx + 1 + self.len) % self.len
        elif keycode == ' ':
            print(f"keycode space, toogle play status")
            self.play = False if self.play else True

    def process(self):
        filepath = self.filepaths[self.idx]
        print(f"frame {self.idx:06d}: {filepath}")
        points = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4) # num_features=4 for kitti
        
        self.cloud_pub.publish(points, stamp=self.get_clock().now().to_msg())

def main(args=None):
    rclpy.init(args=args)

    node = CloudPlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()