from setuptools import setup


setup(
    name='tfrecord',
    version='0.1.0',
    url='https://github.com/leVirve/TFVision.git',
    license='MIT',
    author='Salas leVirve',
    author_email='gae.m.project@gmail.com',
    description='Convert vision data into Tensorflow recommended input `TFRecord`',
    long_description=__doc__,
    py_modules=['tfrecord'],
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
