import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange, SetParametersResult
import os

class DynamicParamNode(Node):
    def __init__(self):
        super().__init__('dynamic_param_node')

        # @! 节点文件中使用 launch 文件中的参数: declare_parameter/declare_parameters 搭配 get_parameter 使用
        # 如果不 declare_parameter 只 get_parameter 会报错未在节点中声明该参数，只 get_parameter_or 不会报错因为它会在参数为声明时使用默认值，但是也无法使用 launch 文件中改动的参数值
        # self.data_root = self.get_parameter_or('data_root', '/datasets/KITTI/object/training/velodyne') # 猜测 get_parameter_or 可能是用于处理函数中动态获取参数时给定一个动态的默认值
        self.declare_parameter('param_str', 'default_str')
        # 设置参数描述符，定义范围，界面显示进度条
        descriptor = ParameterDescriptor()
        range_ = FloatingPointRange()
        range_.from_value, range_.to_value = -100.0, 100.0
        descriptor.floating_point_range.append(range_)
        descriptor.description = "this is a float val"
        self.declare_parameters(
            namespace='',
            parameters=[
                # 基本类型 bool/int/float/str
                ('param_bool', True), 
                ('param_int', 0), 
                ('param_float', 50.0, descriptor),
                # 数组类型，不能动态调参 BOOL_ARRAY/INTEGER_ARRAY/DOUBLE_ARRAY/STRING_ARRAY
                # @! 非常不合理的设计，不能使用空列表 [] 初始化（否则默认会推断为 BYTE_ARRAY），因为它需要根据初始值来判断数据类型，但是又没有提供(参数名,类型,初始值)的三元组
                # 如果值声明参数名和类型，不初始化则不传参时会报错，也不能用两个二元组弄两次，还是会以初始化推断的类型为准，例如 ('param_int_arr', rclpy.Parameter.Type.INTEGER_ARRAY), ('param_int_arr', [1, 2, 3, 4]),
                ('param_int_arr', [1, 2, 3, 4]),
                ('param_str_arr', ['str1', 'str2']), 
                # launch 文件命令行传入数组数据时也一样，必须是传入带数据的字符串如 '[1,2]' 而不能是 '[]'，节点文件声明中也不能是 None [] 之类的
                ('data_list', [1, 2, 3, 4]),        # 如果不需要默认值为空数组，正常使用即可
                ('data_list_default_none', 'None'), # 如果需要则使用字符串类型，再转 Python 代码
                ('choice', 'default_choice')
            ]
        )
        print(self.get_parameter('param_bool').get_parameter_value().bool_value)
        print(self.get_parameter('param_str').value)
        self.param_int_arr = list(self.get_parameter('param_int_arr').get_parameter_value().integer_array_value)
        self.param_str_arr = self.get_parameter('param_str_arr').value # value 为类方法属性会自动转成 Python 类型
        print(self.param_int_arr)
        print(self.param_str_arr)

        self.data_list = self.get_parameter('data_list').value
        #@! 命令行传复杂数据时也只能这样，更安全点的方式是 import ast; ast.literal_eval(list_str)
        self.data_list_default_none = [] if self.get_parameter('data_list_default_none').value == 'None' else eval(self.get_parameter('data_list_default_none').value) # 字符串转代码
        print(self.data_list)
        print(self.data_list_default_none)
        print(self.get_parameter('choice').value)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('param_dict.int_list', [1, 2, 3, 4]),
                ('param_dict.str', "aaabbb"),
            ]
        )
        print(self.get_parameter('param_dict.int_list').value)
        print(self.get_parameter('param_dict.str').value)

        self.timer = self.create_timer(5.0, self.process)
        self.add_on_set_parameters_callback(self.dynamic_param_callback)

    def dynamic_param_callback(self, params):
        mod_keys = [param.name for param in params]
        print("只会传递本次修改的参数: ", mod_keys)
        for param in params:
            if param.name in ['choice']:
                print(f"{param.name} 不可以动态设置")
                return SetParametersResult(successful=False, reason=f'{param.name} can not reconfig') # 置为失败则不会修改参数的值
            if param.name == 'param_str':
                print_str = "param_str is a path" if os.path.exists(param.value) else "param_str is not a path"
                print(print_str)
            print(f"Parameter {param.name} changed to: {param.value}")
        return SetParametersResult(successful=True, reason='success')

    def process(self):
        for key in ['param_bool', 'param_int', 'param_float', 'param_str', 'data_list_default_none', 'choice', 'param_dict.str']:
            print(f"{key}: {self.get_parameter(key).value}")

def main(args=None):
    rclpy.init(args=args)

    node = DynamicParamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()