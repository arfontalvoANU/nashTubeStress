#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as f:
	long_description = f.read()

setuptools.setup(name='nashTubeStress'
	,version='0.0.1'
	,author='Will Logie'
	,author_email='will.logie@anu.edu.au'
	,description="NonAxiSymmetrically Heated (NASH) Tube Stress"
	,long_description=long_description
	,long_description_content_type="text/markdown"
	,url='https://github.com/willietheboy/nashTubeStress'
	,packages=setuptools.find_packages()
	,license="GPLv3, see LICENSE file"
	,classifiers=[
		"Development Status :: 3 - Alpha"
		,"Environment :: Console"
		,"Intended Audience :: Science/Research"
		,"License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
		,"Natural Language :: English"
		,"Operating System :: Microsoft :: Windows :: Windows 10"
		,"Operating System :: POSIX :: Linux"
		,"Programming Language :: Python :: 3"
		,"Topic :: Scientific/Engineering :: Physics"
	]
	,install_requires=['scipy','numpy','matplotlib']
	,include_package_data=True
	,package_data={'': ['mats/*', 'mats/props/*']}
	,python_requires='>=3.8.5'
)

