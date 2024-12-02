from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    data_list = DeclareLaunchArgument('data_list', default_value='[2,3,4,5]', description='')
    data_list_default_none = DeclareLaunchArgument('data_list_default_none', default_value='None', description='')
    choice = DeclareLaunchArgument('choice', default_value='A', description='', choices=['A', 'B'])
    return LaunchDescription([
        data_list, data_list_default_none, choice, 
        Node(
            package='ros2demo', executable='dynamic_param_node', name='dynamic_param_node', output='screen', emulate_tty=True,
            parameters=[
                {'data_list': LaunchConfiguration('data_list')},
                {'data_list_default_none': LaunchConfiguration('data_list_default_none')},
                {'choice': LaunchConfiguration('choice')},
                # {'data_list': [4,5,6,7]}, # 可以在节点参数值像在配置文件中一样传 INTEGER_ARRAY ，但是 DeclareLaunchArgument 中的类型不支持，没法从命令行传列表或者复杂的数据结构
                PathJoinSubstitution([get_package_share_directory('ros2demo'), 'config/demo_param.yaml'])
            ]
        ),
        ExecuteProcess(
            name='rqt_reconfigure',
            cmd=['ros2', 'run', 'rqt_reconfigure', 'rqt_reconfigure'],
        ),
    ])