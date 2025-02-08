from setuptools import setup, find_packages

setup(
    name='shap_value_visualization',
    version='0.1.0',
    description='A production-quality project demonstrating SHAP value visualization for a decision tree model on the Iris dataset',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/shap-value-visualization',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'scikit-learn>=0.24',
        'shap>=0.41',
        'numpy>=1.18',
        'pandas>=1.0',
        'matplotlib>=3.0',
        'joblib>=0.14',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.6',
)
