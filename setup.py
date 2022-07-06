from setuptools import setup

setup(
    name='SkeletonTools',
    version='1.0',
    packages=['src', 'src.skeleton_tools', 'src.skeleton_tools.utils', 'src.skeleton_tools.datasets', 'src.skeleton_tools.pipe_components', 'src.skeleton_tools.openpose_layouts', 'src.skeleton_tools.skeleton_visualization'],
    url='https://github.com/TalBarami/SkeletonTools',
    license='',
    author='TalBarami',
    author_email='talbaramii@gmail.com',
    description='General code for skeleton data manipulation.'
)
