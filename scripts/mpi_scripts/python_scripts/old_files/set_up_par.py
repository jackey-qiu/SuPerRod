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

print 'setting up pars'
mod = model.Model()
config = io.Config()
opt = diffev.DiffEv()
io.load_gx(sys.argv[1],mod,opt,config)
spectra_index=int(sys.argv[2])

#set data use
for item in mod.data.items:
    item.use=False
#here assume the first RAXR spectra is on the second position right after CTR dataset
mod.data.items[spectra_index+1].use=True

#now set pars
row_to_be_set=None
for i in range(mod.parameters.get_len_rows()):
    mod.parameters.set_value(i,2,False)
    if mod.parameters.get_value(i,0)=='rgh_raxs.setA'+str(spectra_index+1):
        row_to_be_set=i
num_pars_each_roll=None
if "'MI'" in mod.get_script():
    num_pars_each_roll=5
else:
    num_pars_each_roll=3
#ensure the run is set to be True
if 'RUN=0' in mod.get_script() or 'RUN=False' in mod.get_script():
    mod.set_script(mod.get_script().replace('RUN=0','RUN=1'))
    mod.set_script(mod.get_script().replace('RUN=False','RUN=1'))
for i in range(num_pars_each_roll):
    mod.parameters.set_value(row_to_be_set+i,2,True)
io.save_gx(sys.argv[1],mod,opt,config)
