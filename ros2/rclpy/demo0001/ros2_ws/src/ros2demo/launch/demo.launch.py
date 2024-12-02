from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # 定义 launch 文件的命令行参数，同 ROS1 的 arg 标签
    data_root = DeclareLaunchArgument('data_root', default_value='/datasets/KITTI/object/training/velodyne', description='.bin 格式点云文件的目录')
    run_rviz = DeclareLaunchArgument('run_rviz', default_value='true', description='Whether to run rviz')

    return LaunchDescription([
        data_root, run_rviz, # 加到命令行参数中，使得可以命令行通过如 run_rviz:=false 进行修改
        Node(
            package='ros2demo', executable='cloud_play_node', name='cloud_play_node1', output='screen', emulate_tty=True,
            parameters=[
                {'data_root': LaunchConfiguration('data_root')}, # or ParameterValue 同 ROS1 的 param 标签
            ]
        ),
        Node(
            condition=IfCondition(LaunchConfiguration('run_rviz')),
            package='rviz2', executable='rviz2', name='rviz2',
            # 注意 get_package_share_directory 拿到的是 share 目录，故运行时保存 rviz 不会改动 src 下的 rviz 文件
            arguments=['-d', PathJoinSubstitution([get_package_share_directory('ros2demo'), 'rviz/demo.rviz'])]
        ),
    ])