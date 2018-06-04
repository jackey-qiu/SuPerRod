#!/usr/bin/python
####read this first###
"""
1. This script is used to refine RAXR data based on nonlinear least square algorithem.
2. It will take one gx file as the single argument using the syntex in a terminal >>python GenX_run_model_lm.py some_GenX_file.gx
3. It will use the best fit GenX results as the starting conditions for the refinement.
4. The boundary used in gx file will be extracted as the bounds used in the LS refinment.
5. Refer to Scripy.curve_fit document for more details about the refinement function
6. Note you need a Scipy newer than 0.18 to run this script.
"""

#set the genx path here according to your sys
import sys
genxpath = 'P://apps//genx_pc_qiu'
sys.path.insert(0,genxpath)

import numpy as np
from datetime import datetime
import model, fom_funcs, diffev
import filehandling as io
from scipy.optimize import curve_fit

# Okay lets make it possible to batch script this file ...
if len(sys.argv) !=2:
    print sys.argv
    print 'Wrong number of arguments to %s'%sys.argv[0]
    print 'Usage: %s infile.gx'%sys.argv[0]
    sys.exit(1)

#gx file you feed in the script
infile = sys.argv[1]
# build outfile name
outfile = infile
outfile = outfile.replace('.gx','')
outfile=outfile+'_lm_ran.gx'
###############################################################################
# Parameter section - modify values according to your needs
###############################################################################
#####################
# figure of merit (FOM) to use, valid foms are in this list['R1','R2','log','diff','sqrt','chi2bars','chibars','logbars','sintth4']
fom = 'chi2bars'
#number of raxr datasets (21 in total usually), or give it a list of data index you want to fit, eg [2,3,4] will fit the 2nd, 3rd and 4th set of RAXR data
N_raxr=21
if type(N_raxr)==int:
    i_container=range(N_raxr)
else:
    i_container=np.array(N_raxr)-1

def autosave():
    #print 'Updating the parameters'
    mod.parameters.set_value_pars(opt.best_vec)
    io.save_gx(outfile, mod, opt, config)


#extract boundary and best vec from genx file
def extract_bounds(par):
    left,right,best_vec=[],[],[]
    rows_num=par.get_len_rows()
    for i in range(rows_num):
        if par.get_value(i,2)==True:
            left.append(par.get_value(i,3))
            right.append(par.get_value(i,4))
            best_vec.append(par.get_value(i,1))
    return (left,right),best_vec

def extract_bounds_2(par,i_data):
    left,right,best_vec,par_index=[],[],[],[]
    rows_num=par.get_len_rows()
    for i in range(rows_num):
        if par.get_value(i,0)=='rgh_raxs.setA'+str(i_data):
            par_raxr=None
            if "\"MI\"" in mod.get_script() or "\'MI\'" in mod.get_script():
                par_raxr=5
            elif "\"MD\"" in mod.get_script() or "\'MD\'" in mod.get_script():
                par_raxr=3
            for j in range(par_raxr):
                left.append(par.get_value(i+j,3))
                right.append(par.get_value(i+j,4))
                best_vec.append(par.get_value(i+j,1))
                par_index.append(i+j)
            break
        else:
            pass
    return (left,right),best_vec,par_index

def extract_xy(data):
    x,y,err,index_use=[],[],[],[]
    for i in range(len(data)):
        each=data[i]
        if each.use:
            x=x+list(each.x)
            y=y+list(each.y)
            err=err+list(each.error)
            index_use.append(i)
    return x,y,err,index_use

def extract_xy_2(data,i):
    x,y,err,index_use=[],[],[],[]
    each=data[i]
    x=x+list(each.x)
    y=y+list(each.y)
    err=err+list(each.error)
    index_use.append(i)
    return x,y,err,index_use

for i in i_container:#looping through 21 RAXR datasets normally
    mod = model.Model()
    config = io.Config()
    opt = diffev.DiffEv()
    if i==0:
        io.load_gx(infile, mod, opt, config)
    else:
        io.load_gx(outfile, mod, opt, config)

    data_sets=mod.data.items
    bounds=[]
    x,y,err,index_use=extract_xy_2(data_sets,i+1)
    bounds,best_vec,par_indx=extract_bounds_2(mod.parameters,i+1)

    #set the data used and the associated pars columns
    for j in range(len(data_sets)):
        if j not in index_use:
            mod.data.items[j].use=False
        else:
            mod.data.items[j].use=True
    for j in range(mod.parameters.get_len_rows()):
        if j in par_indx:
            mod.parameters.set_value(j,2,True)
        else:
            mod.parameters.set_value(j,2,False)

    opt.best_vec=best_vec
    mod.simulate()
    opt.init_fitting(mod)
    eval('mod.set_fom_func(fom_funcs.%s)' % fom)
    opt.set_autosave_func(autosave)
    #set the initial condition as the best_vec
    pars_init=opt.best_vec
    #run number for output purpose during fit going on
    global run_num
    run_num=0
    #fom before lm refinement
    print "Before fit, the fom=",str(opt.calc_fom(opt.best_vec))
    #fit function to extract the fom vectors
    def fit_function(x,*pars):
        opt.best_vec=pars
        #update parameter vectors
        map(lambda func, value:func(value), opt.par_funcs, pars)
        try:
            simulated_data,wt,wt_list = mod.script_module.Sim(mod.data)
            fom_raw, fom_inidv, fom = mod.calc_fom(simulated_data,wt,wt_list)
        except:
            simulated_data,wt = mod.script_module.Sim(mod.data)
            fom_raw, fom_inidv, fom = mod.calc_fom(simulated_data,wt)
        data_flatten=[]
        fom_flatten=[]
        for i in index_use:
            data_flatten=data_flatten+list(simulated_data[i])
            fom_flatten=fom_flatten+list(fom_raw[i]/(len(y)-len(pars)))#fom_raw should be scaled by (N-P) total number of datapoint - par numbers
        global run_num
        run_num+=1
        if not run_num%100:
            print "running on with trial "+str(run_num)+" with fom=",str(fom)
        if not run_num%100:
            print "save temp file"
            autosave()
        return np.sqrt(np.array(fom_flatten)*2)#to make sure cost is the fom value calcuated in GenX
    print "fitting started for RAXR dataset"+str(i+1)
    popt, pcov = curve_fit(fit_function, x, np.array(y)*0.,p0=pars_init,method="trf",loss="linear",bounds=bounds,max_nfev=500,verbose=2)

    #cal the standard deviation of fit parameter values
    perr = np.sqrt(np.diag(pcov))
    print "Model refinement completed sucessfully"
    print 'Fitting results summary'
    print 'par      bestfit value       error'
    #now update the error column in the file
    n_elements = mod.parameters.get_len_rows()
    index_container=[]
    for index in range(n_elements):
        if mod.parameters.get_value(index,2)==True:
            index_container.append(index)
    for k in range(len(perr)):
        mod.parameters.set_value(index_container[k],5,str(perr[k]))
        print mod.parameters.get_value(index_container[k],0),mod.parameters.get_value(index_container[k],1),mod.parameters.get_value(index_container[k],5)
    autosave()
