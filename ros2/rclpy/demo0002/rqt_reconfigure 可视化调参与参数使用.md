#### rqt_reconfigure 可视化调参与参数使用 [ [1](https://blog.csdn.net/2203_76027118/article/details/136740657)* ]
```bash
# 参考链接中的 C++ 代码修改
# ROS2 中不论动静态参数都可以直接 declare_parameter
# ros2 run ros2demo dynamic_param_node
ros2 run ros2demo dynamic_param_node --ros-args --params-file src/ros2demo/config/demo_param.yaml
ros2 run rqt_reconfigure rqt_reconfigure

ros2 launch ros2demo demo.launch.py
ros2 run ros2demo teleop_key_node
```