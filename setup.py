from setuptools import setup
import setuptools
import pip
import os

from pip._internal.req import parse_requirements
from pathlib import Path
this_directory = Path(__file__).parent

links = []
requires = []

requirements = parse_requirements('requirements.txt', session='hack')
requirements = list(requirements) 
try:
    requirements = [str(ir.req) for ir in requirements]
except:
    requirements = [str(ir.requirement) for ir in requirements]

setup(name='spt_align',
			version='1.0',
			description='A package to perform SPT-based subpixel image registration',
			long_description=(this_directory / "README.md").read_text(),
			#long_description=open('README.rst',encoding="utf8").read(),
			long_description_content_type='text/markdown',
			url='http://github.com/remyeltorro/SPTAlign',
			author='Rémy Torro',
			author_email='remy.torro@gmail.com',
			license='GPL-3.0',
			packages=setuptools.find_packages(),
			zip_safe=False,
			package_data={'spt_align': ['*']},
			install_requires = requirements,
			#dependency_links = links
			)

