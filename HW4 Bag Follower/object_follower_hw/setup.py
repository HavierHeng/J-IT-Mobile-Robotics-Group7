from setuptools import setup

package_name = 'object_follower_hw'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='1006027@mymail.sutd.edu.sg',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'straight_follower = object_follower_hw.object_follower:main',
            'chase_follower = object_follower_hw.object_follower_enhanced:main'
        ],
    },
)
