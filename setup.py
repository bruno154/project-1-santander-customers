from setuptools import find_packages, setup

install_requires = [
    'pandas>=0.25.0',
    'numpy>=1.15.4',
    'glob2>=0.6',
    'functools'
]

# tests_require = ['pytest>=4.0.2']

setup(name='Satisfaction',
      version='0.0.1',
      description='Projeto para classificação do nível de satisfação de clientes',
      author='Bruno Vinicius Nonato',
      author_email='brunovinicius154@gmail.com',
      install_requires=install_requires,
      packages=['Satisfaction'])