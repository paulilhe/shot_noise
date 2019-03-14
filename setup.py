from numpy.distutils.core import setup, Extension

setup(name='shot_noise',
      version='0.1',
      description='This package is related to the simulation and the estimation of the statistical elements of shot-noise processes based on the two estimation techniques introduced in the paper Nonparametric estimation of shot-noise processes',
      url='https://github.com/dmikushin/shot_noise',
      author='Paul Ilhe',
      license='MIT',
      packages=['shot_noise'],
      ext_modules=[Extension(name='design_operation', sources=['shot_noise/matrix_operations.f90'], language='f90')],
      install_requires=['numpy', 'pp', 'patsy', 'scipy', 'cvxopt'],
      zip_safe=False)

