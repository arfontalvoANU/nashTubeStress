#!/usr/bin/env python3
# Copyright (C) 2021 William R. Logie

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""
nashTubeStress.py
 -- steady-state temperature distribution (Gauss-Seidel iteration)
 -- biharmonic thermoelastic stress

See also:
 -- Solar Energy 160 (2018) 368-379
 -- https://doi.org/10.1016/j.solener.2017.12.003
"""

import sys, time, os
from math import exp, log, sqrt, pi, ceil, floor, asin

import numpy as np
import scipy.optimize as opt
from pint import UnitRegistry
UR_ = UnitRegistry()
Q_ = UR_.Quantity

import nashTubeStress as nts
import coolant
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

################################### PLOTTING ###################################

import matplotlib as mpl
import matplotlib.pyplot as plt
## uncomment following if not in ~/.config/matplotlib/matplotlibrc already
plt.rc('text', usetex = True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{newtxtext,newtxmath,siunitx}"
mpl.rc('figure.subplot', bottom=0.13, top=0.95)
mpl.rc('figure.subplot', left=0.15, right=0.95)
from matplotlib import colors, ticker, cm
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.projections import PolarAxes
import matplotlib.transforms as mtransforms
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
from mpl_toolkits.axisartist.grid_finder import \
	(FixedLocator, MaxNLocator, DictFormatter)

def plotTemperatureAnnotate(theta, r, T, TMin, TMax, filename):
	fig = plt.figure(figsize=(3, 3.25))
	fig.subplots_adjust(left=-1)
	fig.subplots_adjust(right=1)
	fig.subplots_adjust(bottom=0.1)
	fig.subplots_adjust(top=0.9)
	ax = fig.add_subplot(111, projection='polar')
	ax.set_theta_direction(-1)
	ax.set_theta_offset(np.radians(90))
	#cmap = cmaps.magma # magma, inferno, plasma, viridis...
	cmap = cm.get_cmap('magma')
	levels = ticker.MaxNLocator(nbins=10).tick_values(TMin-273.15, TMax-273.15)
	cf = ax.contourf(theta, r, T-273.15, levels=levels, cmap=cmap)
	ax.set_rmin(0)
	cb = fig.colorbar(cf, ax=ax)
	cb.set_label(r'\textsc{temperature}, $T$ (\si{\celsius})')
	ax.patch.set_visible(False)
	ax.spines['polar'].set_visible(False)
	gridlines = ax.get_xgridlines()
	ticklabels = ax.get_xticklabels()
	for i in range(5, len(gridlines)):
		gridlines[i].set_visible(False)
		ticklabels[i].set_visible(False)
	ax.annotate(r'\SI{'+'{0:.0f}'.format(T.max()-273.15)+'}{\celsius}', \
				 xy=(theta[0,-1], r[0,-1]), \
				 xycoords='data', xytext=(40, 10), \
				 textcoords='offset points', fontsize=12, \
				 arrowprops=dict(arrowstyle='->'))
	ax.grid(axis='y', linewidth=0)
	ax.grid(axis='x', linewidth=0.2)
	plt.setp(ax.get_yticklabels(), visible=False)
	fig.savefig(filename, transparent=True)
	plt.close(fig)

def plotStress(theta, r, sigma, sigmaMin, sigmaMax, filename):
	fig = plt.figure(figsize=(3.5, 3.5))
	fig.subplots_adjust(left=-1)
	fig.subplots_adjust(right=1)
	fig.subplots_adjust(bottom=0.1)
	fig.subplots_adjust(top=0.9)
	ax = fig.add_subplot(111, projection='polar')
	ax.set_theta_direction(-1)
	ax.set_theta_offset(np.radians(90))
	cmap = cm.get_cmap('magma')
	#cmap = cmaps.magma
	levels = ticker.MaxNLocator(nbins=10).tick_values(
		sigmaMin*1e-6, sigmaMax*1e-6
	)
	cf = ax.contourf(theta, r, sigma*1e-6, levels=levels, cmap=cmap)
	ax.set_rmin(0)
	cb = fig.colorbar(cf, ax=ax)
	cb.set_label('$\sigma$ [MPa]')
	ax.patch.set_visible(False)
	ax.spines['polar'].set_visible(False)
	gridlines = ax.get_xgridlines()
	ticklabels = ax.get_xticklabels()
	for i in range(5, len(gridlines)):
		gridlines[i].set_visible(False)
		ticklabels[i].set_visible(False)
	ax.grid(axis='y', linewidth=0)
	ax.grid(axis='x', linewidth=0.2)
	plt.setp(ax.get_yticklabels(), visible=False)
	fig.savefig(filename, transparent=True)
	plt.close(fig)

def plotStressAnnotate(theta, r, sigma, sigmaMin, sigmaMax, annSide, filename):
	annSide = -70 if annSide=='left' else 40
	fig = plt.figure(figsize=(3, 3.25))
	fig.subplots_adjust(left=-1)
	fig.subplots_adjust(right=1)
	fig.subplots_adjust(bottom=0.1)
	fig.subplots_adjust(top=0.9)
	ax = fig.add_subplot(111, projection='polar')
	ax.set_theta_direction(-1)
	ax.set_theta_offset(np.radians(90))
	cmap = cm.get_cmap('magma')
	#sigmaMin = np.min(sigma*1e-6); sigmaMax = np.max(sigma*1e-6)
	levels = ticker.MaxNLocator(nbins=10).tick_values(sigmaMin*1e-6, sigmaMax*1e-6)
	cf = ax.contourf(theta, r, sigma*1e-6, levels=levels, cmap=cmap)
	ax.set_rmin(0)
	cb = fig.colorbar(cf, ax=ax)
	cb.set_label(r'\textsc{equivalent stress}, $\sigma_\mathrm{eq}$ (MPa)')
	ax.patch.set_visible(False)
	ax.spines['polar'].set_visible(False)
	gridlines = ax.get_xgridlines()
	ticklabels = ax.get_xticklabels()
	for i in range(5, len(gridlines)):
		gridlines[i].set_visible(False)
		ticklabels[i].set_visible(False)
	#annInd = np.unravel_index(s.sigmaEq.argmax(), s.sigmaEq.shape)
	annInd = (0, -1)
	ax.annotate('\SI{'+'{0:.0f}'.format(np.max(sigma*1e-6))+'}{\mega\pascal}', \
				 xy=(theta[annInd], r[annInd]), \
				 xycoords='data', xytext=(annSide, 10), \
				 textcoords='offset points', fontsize=12, \
				 arrowprops=dict(arrowstyle='->'))
	ax.grid(axis='y', linewidth=0)
	ax.grid(axis='x', linewidth=0.2)
	plt.setp(ax.get_yticklabels(), visible=False)
	fig.savefig(filename, transparent=True)
	plt.close(fig)

def plotComponentStress(r, sigmaR, sigmaTheta, sigmaZ,
						sigmaEq, filename, i, loc):
	a = r[0,0]; b = r[0,-1]
	trX = Q_(1, 'inch').to('mm').magnitude
	trY = Q_(1, 'ksi').to('MPa').magnitude
	trans = mtransforms.Affine2D().scale(trX,trY)
	fig = plt.figure(figsize=(4, 3.5))
	ax = SubplotHost(fig, 1, 1, 1)
	axa = ax.twin(trans)
	axa.set_viewlim_mode("transform")
	axa.axis["top"].set_label(r'\textsc{radius}, $r$ (in.)')
	axa.axis["top"].label.set_visible(True)
	axa.axis["right"].set_label(r'\textsc{stress component}, $\sigma$ (ksi)')
	axa.axis["right"].label.set_visible(True)
	ax = fig.add_subplot(ax)
	ax.plot(r[i,:]*1e3, sigmaR[i,:]*1e-6, '^-',
			label='$\sigma_r$')
	ax.plot(r[i,:]*1e3, sigmaTheta[i,:]*1e-6, 'o-',
			label=r'$\sigma_\theta$')
	ax.plot(r[i,:]*1e3, sigmaZ[i,:]*1e-6, 'v-',
			label='$\sigma_z$')
	ax.plot(r[i,:]*1e3, sigmaEq[i,:]*1e-6, 's-',
			label='$\sigma_\mathrm{eq}$')
	ax.set_xlabel(r'\textsc{radius}, $r$ (mm)')
	ax.set_xlim((a*1e3)-0.1,(b*1e3)+0.1)
	ax.set_ylabel(r'\textsc{stress component}, $\sigma$ (MPa)')
	ax.legend(loc=loc)
	#labels = ax.get_xticklabels()
	#plt.setp(labels, rotation=30)
	fig.tight_layout()
	fig.savefig(filename, transparent=True)
	plt.close(fig)

################################### FUNCTIONS ##################################

def headerprint(string, mychar='='):
	""" Prints a centered string to divide output sections. """
	mywidth = 64
	numspaces = mywidth - len(string)
	before = int(ceil(float(mywidth-len(string))/2))
	after  = int(floor(float(mywidth-len(string))/2))
	print("\n"+before*mychar+string+after*mychar+"\n")

def valprint(string, value, unit='-'):
	""" Ensure uniform formatting of scalar value outputs. """
	print("{0:>30}: {1: .4f} {2}".format(string, value, unit))

def valeprint(string, value, unit='-'):
	""" Ensure uniform formatting of scalar value outputs. """
	print("{0:>30}: {1: .4e} {2}".format(string, value, unit))

def matprint(string, value):
	""" Ensure uniform formatting of matrix value outputs. """
	print("{0}:".format(string))
	print(value)

def findFlux(flux, s, f, i, point):
	"""
	Helper for finding optimum flux for certain stress condition
	"""
	s.CG = flux

	ret = s.solve(eps=1e-6)
	s.postProcessing()

	if point=='max':
		# T_max|sigmaEqMax:
		sigmaEqMax = np.interp(np.max(s.T), f[:,0], f[:,i])
		return sigmaEqMax - np.max(s.sigmaEq)
	elif point=='inside':
		# T_i:
		sigmaEqMax = np.interp(s.T[0,0], f[:,0], f[:,i])
		return sigmaEqMax - s.sigmaEq[0,0]
	elif point=='outside':
		# T_o
		sigmaEqMax = np.interp(s.T[0,-1], f[:,0], f[:,i])
		return sigmaEqMax - s.sigmaEq[0,-1]
	elif point=='membrane':
		# Assuming shakedown has occured (membrane stress remains):
		sigmaEqMax = np.interp(np.average(s.T[0,:]), f[:,0], f[:,i])
		return sigmaEqMax - np.average(s.sigmaEq[0,:])
	elif point=='corrosion':
		# T_i:
		return 630 + 273.15 - s.T[0,0]
	else: sys.exit('Variable point {} not recognised'.format(point))

##################################### MAIN #####################################

if __name__ == "__main__":
	h_ext=10. # convective loss due to wind W/(m2.K)
	salt = coolant.nitrateSalt(False); salt.update(723.15)
	iterator='inline'
	nr=6; nt=61

	T_int = np.linspace(290, 565, 12)
	T_int = np.append(T_int,600) + 273.15

	""" Looping velocities"""
	for OD,WT in zip([45],[1.5]):
		""" Instantiating figure and subplots"""
		fig = plt.figure(figsize=(3.5, 3.5))
		ax = fig.add_subplot(111)
		csv = np.c_[T_int,]
		header  = '0'

		b = OD/2e3		 # outside tube radius [mm->m]
		a = (b-WT*1e-3)	 # inside tube radius [mm->m]
		g = nts.Grid(nr=nr, nt=nt, rMin=a, rMax=b) # nr, nt -> resolution
		vfs = np.array([0, 0.25, 0.5, 0.75, 1., 2., 3., 4.])

		fname = 'N08810'
		for vf in vfs[1:]:
			for mat in ['800H']:
				props = np.genfromtxt(os.path.join('mats/props', mat), delimiter=';')
				props[:,0] += 273.15 # degC to K
				props[:,1] *= 1e9  # convert GPa -> Pa
				props[:,3] *= 1e-6 # convert 1e-6 mm/mm -> mm/mm
				E     = np.interp(723.15,props[:,0],props[:,1])
				k     = np.interp(723.15,props[:,0],props[:,2])
				alpha = np.interp(723.15,props[:,0],props[:,3])
				nu = 0.31
				s = nts.Solver(g, debug=False, CG=0.85e6, k=k, T_int=723.15, R_f=8.808e-5,
								A=0.968, epsilon=0.87, T_ext=293.15, h_ext=h_ext,
								P_i=0e5, alpha=alpha, E=E, nu=nu, n=1,
								bend=False)
				s.extBC = s.extTubeHalfCosFluxRadConv
				s.intBC = s.intTubeConv
				fv = np.genfromtxt(os.path.join('mats', mat), delimiter=';')
				fv[:,0] += 273.15 # degC to K
				fv[:,2] *= 3e6 # apply 3f criteria to Sm and convert MPa->Pa
				TSod_met = np.zeros(len(T_int))
				fluxSalt = np.zeros(len(T_int))
				print('vf: {0:.2f} m/s'.format(vf))
				for i in tqdm(range(len(T_int))):
					s.E =     np.interp(T_int[i],props[:,0],props[:,1])
					s.k =     np.interp(T_int[i],props[:,0],props[:,2])
					s.alpha = np.interp(T_int[i],props[:,0],props[:,3])
					s.T_int = T_int[i]
					salt.update(T_int[i])
					s.h_int, dP = coolant.HTC(False, salt, a, b, s.k, 'Gnielinski', 'velocity', vf)
					fluxSalt[i] = opt.newton(
						findFlux, 1e5,
						args=(s, fv, 2, 'outside'),
						maxiter=1000, tol=1e-2
					)
					TSod_met[i] = np.max(s.T)

				ax.plot(T_int-273.15,fluxSalt*1e-6, label=r'U = {0} m/s'.format(vf))
				ax.set_xlabel(r'\textsc{fluid temperature}, '+\
							  '$T_\mathrm{f}$ (\si{\celsius})')
				ax.set_ylabel(
					r'\textsc{incident flux}, $\vec{\phi_\mathrm{q}}$ '+\
					'(\si{\mega\watt\per\meter\squared})'
				)
				ax.legend(loc='best')
				fig.tight_layout()
				fig.savefig('{0}_OD{1:.2f}_WT{2:.2f}_peakFlux.pdf'.format(fname, OD, WT),
							transparent=True)
				plt.close(fig)
				## Dump peak flux results to CSV file:
				csv  = np.c_[csv, fluxSalt,]

		csv = np.concatenate((vfs.reshape(1,len(vfs)),csv))
		np.savetxt('{0}_OD{1:.2f}_WT{2:.2f}_peakFlux.csv'.format(fname, OD, WT), csv, delimiter=',')
