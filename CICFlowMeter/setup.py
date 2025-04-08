from setuptools import setup, find_packages

setup(
    name="CICFlowMeter",
    version="0.1",
    author="Your Name",
    description="A Python implementation of CICFlowMeter for network traffic analysis",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'scapy>=2.4.5',
        'numpy>=1.19.5',
        'pandas>=1.3.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: System :: Networking :: Monitoring",
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'cicflowmeter=CICFlowMeter.CICFlowMeter.main:main',
        ],
    }
)