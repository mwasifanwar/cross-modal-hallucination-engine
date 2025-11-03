from setuptools import setup, find_packages

setup(
    name="cross-modal-hallucination-engine",
    version="1.0.0",
    author="mwasifanwar",
    description="Advanced AI system for generating missing modalities from available data using cross-modal hallucination",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "pillow>=9.0.0",
        "opencv-python>=4.7.0",
        "librosa>=0.10.0",
        "pytest>=7.3.0"
    ],
    python_requires=">=3.8",
)