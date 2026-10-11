from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class MultiBoxLayer(nn.Module):
	num_anchors = [6,6,4,4,4,4,4]
	
	def __init__(self,num_classes):
		super(MultiBoxLayer, self).__init__()
		self.num_classes = num_classes
		self.in_planes = [512,256,256,256,256,512,1024]
		self.loc_layers = nn.ModuleList()
		self.conf_layers = nn.ModuleList()
		for i in range(len(self.in_planes)):
			self.loc_layers.append(nn.Conv2d(self.in_planes[i],
											 self.num_anchors[i]*4,
											 kernel_size=3, padding=1))
			self.conf_layers.append(nn.Conv2d(self.in_planes[i],
											  self.num_anchors[i]*self.num_classes,
											  kernel_size=3, padding=1))
		self.softmax = nn.Softmax(dim = -1)
	def forward(self, xs):
		'''
		Args:
		  xs: (list) of tensor containing intermediate layer outputs.

		Returns:
		  loc_preds: (tensor) predicted locations, sized [N,8732,4].
		  conf_preds: (tensor) predicted class confidences, sized [N,8732,21].
		'''
		y_locs = []
		y_confs = []
		for i, x in enumerate(xs):
			y_loc = self.loc_layers[i](x)
			"""
				loc.shape: torch.Size([1, 24, 10, 10])
				loc.shape: torch.Size([1, 24, 5, 5])
				loc.shape: torch.Size([1, 16, 3, 3])
				loc.shape: torch.Size([1, 16, 1, 1])
				loc.shape: torch.Size([1, 16, 75, 75])
				loc.shape: torch.Size([1, 16, 38, 38])
				loc.shape: torch.Size([1, 16, 19, 19])
			"""
			# print('loc.shape: {}'.format(y_loc.size()))
			N = y_loc.size(0)
			y_loc = y_loc.permute(0,2,3,1).contiguous()
			y_loc = y_loc.view(N,-1,4)
			y_locs.append(y_loc)

			y_conf = self.conf_layers[i](x)
			y_conf = y_conf.permute(0,2,3,1).contiguous()
			y_conf = y_conf.view(N,-1,self.num_classes)
			y_confs.append(y_conf)

		loc_preds = torch.cat(y_locs, 1)
		conf_preds = torch.cat(y_confs, 1)

		if self.training:
			outputs = (loc_preds, conf_preds)
		else:
			conf_preds = self.softmax(conf_preds)
			outputs = (loc_preds,conf_preds)

		return outputs