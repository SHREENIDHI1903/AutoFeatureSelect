from setuptools import setup, find_packages

setup(
    name='autofeatureselect',
    version='0.1.0',
    author='Shreenidhi T H',
    author_email='shreenidhi1903@gmail.com',
    description='A simple feature selection package for ML',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/autofeatureselect',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'shap'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
