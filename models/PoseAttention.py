import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import models.modules.PoseAttention as M

class HourglassAttention(nn.Module):
	"""docstring for HourglassAttention"""
	def __init__(self, nChannels = 256, numReductions = 4, nModules = 2, poolKernel = (2,2), poolStride = (2,2), upSampleKernel = 2):
		super(HourglassAttention, self).__init__()
		self.numReductions = numReductions
		self.nModules = nModules
		self.nChannels = nChannels
		self.poolKernel = poolKernel
		self.poolStride = poolStride
		self.upSampleKernel = upSampleKernel
		"""
		For the skip connection, a Residual module (or sequence of residuaql modules)
		"""

		_skip = []
		for _ in range(self.nModules):
			_skip.append(M.Residual(self.nChannels, self.nChannels))

		self.skip = nn.Sequential(*_skip)

		"""
		First pooling to go to smaller dimension then pass input through
		Residual Module or sequence of Modules then  and subsequent cases:
			either pass through Hourglass of numReductions-1
			or pass through Residual Module or sequence of Modules
		"""

		self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)

		_afterpool = []
		for _ in range(self.nModules):
			_afterpool.append(M.Residual(self.nChannels, self.nChannels))

		self.afterpool = nn.Sequential(*_afterpool)

		if (numReductions > 1):
			self.hg = HourglassAttention(self.nChannels, self.numReductions-1, self.nModules, self.poolKernel, self.poolStride)
		else:
			_num1res = []
			for _ in range(self.nModules):
				_num1res.append(M.Residual(self.nChannels,self.nChannels))

			self.num1res = nn.Sequential(*_num1res)  # doesnt seem that important ?

		"""
		Now another Residual Module or sequence of Residual Modules
		"""

		_lowres = []
		for _ in range(self.nModules):
			_lowres.append(M.Residual(self.nChannels,self.nChannels))

		self.lowres = nn.Sequential(*_lowres)

		"""
		Upsampling Layer (Can we change this??????)
		As per Newell's paper upsamping recommended
		"""
		self.up = nn.Upsample(scale_factor = self.upSampleKernel)


	def forward(self, x):
		out1 = x
		out1 = self.skip(out1)
		out2 = x
		out2 = self.mp(out2)
		out2 = self.afterpool(out2)
		if self.numReductions>1:
			out2 = self.hg(out2)
		else:
			out2 = self.num1res(out2)
		out2 = self.lowres(out2)
		out2 = self.up(out2)

		return out2 + out1


class PoseAttention(nn.Module):
	"""docstring for PoseAttention"""
	def __init__(self, nChannels, nStack, nModules, numReductions, nJoints, LRNSize, IterSize):
		super(PoseAttention, self).__init__()
		self.nChannels = nChannels
		self.nStack = nStack
		self.nModules = nModules
		self.numReductions = numReductions
		self.nJoints = nJoints
		self.LRNSize = LRNSize
		self.IterSize = IterSize
		self.nJoints = nJoints

		self.start = M.BnReluConv(3, 64, kernelSize = 7, stride = 2, padding = 3)

		self.res1 = M.Residual(64, 128)
		self.mp = nn.MaxPool2d(2, 2)
		self.res2 = M.Residual(128, 128)
		self.res3 = M.Residual(128, self.nChannels)

		_hourglass, _Residual, _lin1, _attiter, _chantojoints, _lin2, _jointstochan = [], [],[],[],[],[],[]

		for i in range(self.nStack):
			_hourglass.append(HourglassAttention(self.nChannels, self.numReductions, self.nModules))
			_ResidualModules = []
			for _ in range(self.nModules):
				_ResidualModules.append(M.Residual(self.nChannels, self.nChannels))
			_ResidualModules = nn.Sequential(*_ResidualModules)
			_Residual.append(_ResidualModules)
			_lin1.append(M.BnReluConv(self.nChannels, self.nChannels))
			_attiter.append(M.AttentionIter(self.nChannels, self.LRNSize, self.IterSize))
			if i<self.nStack//2:
				_chantojoints.append(
						nn.Sequential(
							nn.BatchNorm2d(self.nChannels),
							nn.Conv2d(self.nChannels, self.nJoints,1),
						)
					)
			else:
				_chantojoints.append(M.AttentionPartsCRF(self.nChannels, self.LRNSize, self.IterSize, self.nJoints))
			_lin2.append(nn.Conv2d(self.nChannels, self.nChannels,1))
			_jointstochan.append(nn.Conv2d(self.nJoints,self.nChannels,1))

		self.hourglass = nn.ModuleList(_hourglass)
		self.Residual = nn.ModuleList(_Residual)
		self.lin1 = nn.ModuleList(_lin1)
		self.chantojoints = nn.ModuleList(_chantojoints)
		self.lin2 = nn.ModuleList(_lin2)
		self.jointstochan = nn.ModuleList(_jointstochan)

	def forward(self, x):
		x = self.start(x)
		x = self.res1(x)
		#print("1", x.mean())
		x = self.mp(x)
		x = self.res2(x)
		#print("2", x.mean())
		x = self.res3(x)
		out = []

		for i in range(self.nStack):
			#print("3", x.mean())
			x1 = self.hourglass[i](x)
			#print("4", x1.mean())
			x1 = self.Residual[i](x1)
			#print("5", x1.mean())
			x1 = self.lin1[i](x1)
			#print("6", x1.mean())
			out.append(self.chantojoints[i](x1))
			x1 = self.lin2[i](x1)
			#print("7", x1.mean())
			x = x + x1 + self.jointstochan[i](out[i])

		return (out)
