from setuptools import setup, find_packages

setup(name='parallax_simulator_pipeline',
	version='0.0a',
	packages=find_packages(),
	include_package_data=True,
	install_requires=[
		'numpy',
		'pandas',
		'iminuit',
		'scipy',
		'numba']
)