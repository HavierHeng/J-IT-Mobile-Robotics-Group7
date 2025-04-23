from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import EnvironmentVariable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    map_file = PathJoinSubstitution(
        [FindPackageShare('estimation_and_vision_assignment'),
        'resources',
        'map.pkl'],
    )
    return LaunchDescription([
        Node(
            package='estimation_and_vision_assignment',
            executable='monte_carlo_localization.py',
            name='monte_carlo_localization',
            output='screen',
            parameters=[
                
                {'map_file': map_file},
                {'dynamics_translation_noise_std_dev': 0.05},
                {'dynamics_orientation_noise_std_dev': 0.01},
                {'beam_range_measurement_noise_std_dev': 0.9} #0.3
            ],
        ),
        
    ])

