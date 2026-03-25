from setuptools import find_packages, setup

package_name = 'tb3_localization_validation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jvang',
    maintainer_email='johnnyjvang@gmail.com',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'initial_pose_response = tb3_localization_validation.initial_pose_response:main',
            'amcl_pose_stability = tb3_localization_validation.amcl_pose_stability:main',
            'global_local_consistency = tb3_localization_validation.global_local_consistency:main',
            # Added to print and reset json output
            'reset_results = tb3_tf_validation.reset_results:main',
            'summary_report = tb3_tf_validation.summary_report:main',
        ],
    },
)
