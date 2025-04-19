from setuptools import setup

package_name = 'track_follower_challenge'

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
    maintainer_email='torvalds@linux-foundation.org',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'track_follower = track_follower_challenge.track_follower:main',
            'track_follower_cv = track_follower_challenge.track_follower_cv:main',
            'aruco_relay = track_follower_challenge.aruco_relay:main',
        ],
    },
)
