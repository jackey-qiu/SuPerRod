#-*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:29:10 2015

@author: jackey
"""

import os,sys
sys.path.append('C:\\Users\\jackey\\Documents\\GitHub\\genx_pc_qiu')
sys.path.append('C:\\apps\\genx_pc_qiu')
sys.path.append('/u1/uaf/cqiu/genx_pc_qiu')
sys.path.append('/home/qiu05/genx_pc_qiu')

import model,diffev,time,fom_funcs
import filehandling as io
import glob
import numpy as np

mod = model.Model()
config = io.Config()
opt = diffev.DiffEv()
io.load_gx(sys.argv[1],mod,opt,config)
num_pars_each_roll=None
if 'MI' in mod.get_script():
    num_pars_each_roll=5
else:
    num_pars_each_roll=3
for i in range(num_pars_each_roll):
    mod.parameters.set_value(int(sys.argv[2])+i,2,not mod.parameters.get_value(int(sys.argv[2])+i,2))
mod.data.toggle_use(int(sys.argv[3]))
io.save_gx(sys.argv[1],mod,opt,config)
