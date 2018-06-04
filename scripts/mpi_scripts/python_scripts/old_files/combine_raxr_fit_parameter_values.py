# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:29:10 2015

@author: jackey
"""

import os,sys
sys.path.append('C:\\Users\\jackey\\Documents\\GitHub\\genx_pc_qiu')
sys.path.append('C:\\apps\\genx_pc_qiu')
sys.path.append('C:\\apps\\genx_pc_qiu\\lib')
sys.path.append('/u1/uaf/cqiu/genx_pc_qiu')

import model,diffev,time,fom_funcs
import filehandling as io
import glob
import numpy as np

first_grid=54
grid_gap=20
path='/import/c/w/cqiu/temp_gx_files/'
gx_files=glob.glob(path+'*_May8.gx')#file must be sortable to have the code work correctly
gx_files.sort()
mod = model.Model()
config = io.Config()
opt = diffev.DiffEv()
io.load_gx(gx_files[0],mod,opt,config)
for gx_file in gx_files[1:]:
    print "processing ",gx_file
    i=gx_files.index(gx_file)
    begin_grid=first_grid+grid_gap*i
    mod_temp = model.Model()
    config_temp = io.Config()
    opt_temp = diffev.DiffEv()
    io.load_gx(gx_file,mod_temp,opt_temp,config_temp)
    for grid_index in range(begin_grid,begin_grid+grid_gap):
        for k in range(5):
            mod.parameters.set_value(grid_index,k,mod_temp.parameters.get_value(grid_index,k))
io.save_gx(path+"combined_model_file.gx",mod,opt,config)
