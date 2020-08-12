from setuptools import setup

setup(name='gym_coach_vss',
      version='0.0.1',
      install_requires=['gym', 'gym[atari]', 'protobuf',
                        'pyzmq', 'joblib', 'sslclient']
      )
