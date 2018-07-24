#setup.py
from distutils.core import setup

setup(
	name = 'smtpkg7',
	version = '1.0.0',
	description = "about smart phone",
	author = 'gop',
	packages = ['smtpkg7', 'smtpkg7/camera', 'smtpkg7/phone', 'smtpkg7/subcam'],
	py_modules = ['hello']
	)