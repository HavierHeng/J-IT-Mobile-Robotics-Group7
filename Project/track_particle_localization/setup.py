from setuptools import setup

package_name = 'track_particle_localization'

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
            'amcl = track_particle_localization.track_localization_amcl:main',
            'mapped = track_particle_localization.track_localization_rtabmapl:main',
            'slam = track_particle_localization.track_localization_rtabmap_slam:main',
            'get_map_size = track_particle_localization.rtabmap_size:main'
        ],
    },
)
