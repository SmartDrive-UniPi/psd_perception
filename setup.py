from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'psd_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Include config files
        (os.path.join('share', package_name, 'config'), 
            glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'config'), 
            glob(os.path.join('config', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Edoardo Caciorgna',
    maintainer_email='edo.ca1999@gmail.com',
    description='YOLOv11 TensorRT cone detection for ZED2 camera (with also LiDAR data for pose correction)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node = psd_perception.camera_node:main',
			'camera_lidar_node = psd_perception.camera_lidar_node:main',
        ],
    },
)
