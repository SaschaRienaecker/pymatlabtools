from setuptools import setup, find_packages

setup(
    name='pymatlabtools',
    version='1.0.0',
    author='SR',
    description="A Python library designed to simplify the interaction between python and matlab usage.",
    packages=find_packages(),  # Automatically discover and include all packages
    install_requires=[
        # List the dependencies your package needs
        'numpy',
        'scipy',
        'pandas',
    ],
    entry_points={
        'console_scripts': [
            # Define any command-line scripts your package provides
            # 'your_script_name=your_package.module:function_name',
        ],
    },
)
