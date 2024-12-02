#### 发布点云并编写 launch 文件：[ [1](https://github.com/HaiderAbasi/ROS2-Path-Planning-and-Maze-Solving/blob/master/path_planning_ws/src/maze_bot/maze_bot/maze_solver.py)* [2](https://github.com/ros2/examples/blob/rolling/rclpy/topics/pointcloud_publisher/examples_rclpy_pointcloud_publisher/pointcloud_publisher.py) | [3](https://blog.csdn.net/qq_36372352/article/details/135402532) [4](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Using-Parameters-In-A-Class-Python.html) [5](https://roboticsbackend.com/rclpy-params-tutorial-get-set-ros2-params-with-python/)* ]
```bash
rviz2 -d src/ros2demo/rviz/demo.rviz
ros2 run ros2demo cloud_play_node
ros2 run ros2demo teleop_key_node

# or 
ros2 launch ros2demo demo.launch.py data_root:=/path/to/.bin/dir
ros2 run ros2demo teleop_key_node
```
#### VSCode 调试 [ [1](https://github.com/ms-iot/vscode-ros/issues/872) ]
```bash
# VSCode 调试 launch 文件
# 安装微软官方的 ROS 插件，打开项目后选择合适的 Python 解释器如 /usr/bin/python3
# 主侧边栏三角形调试按钮中点击显示所有自动调试配置，创建 ROS: Launch 调试(提前 colcon build 否则选不到 ros2demo 包)
# 此时会在 .vscode 下生成配置文件（没生成可以多试几次或者手动添加配置），修改 target 字段为 install 目录下的 lauch 文件的路径，启动调试
# 注意！！！断点应该打在 install 目录下的源码文件上，而不是 src 下的源码文件，位置例如 install/ros2demo/lib/python3.10/site-packages/ros2demo/cloud_play_node.py

# VSCode 调试节点
# launch 文件手动添加调试当前 python 文件的配置，点开要调试的节点文件，启动调试
# 注意！！！此时的断点打当前打开的文件中（这个打开的文件可以是 src 源码中的文件），不过该文件调用的文件还是 install 下打包的库代码中的文件
```