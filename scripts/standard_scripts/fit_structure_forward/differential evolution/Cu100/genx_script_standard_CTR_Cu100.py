import models.sxrd as model
from models.utils import UserVars
from datetime import datetime
import numpy as np
import sys,pickle,__main__,os
import batchfile.locate_path as batch_path
import dump_files.locate_path as output_path
import models.domain_creator as domain_creator
import accessory_functions.make_par_table.make_parameter_table_GenX_hematite_rcut as make_grid
import accessory_functions.data_formating.formate_xyz_to_vtk as xyz
from copy import deepcopy
import models.setup_domain_hematite_rcut as setup_domain_hematite_rcut


COUNT_TIME=False
if COUNT_TIME:t_0=datetime.now()

#setting slabs##
wal=0.8625#wavelength of x ray
unitcell = model.UnitCell(3.615, 3.615, 3.615, 90, 90, 90)
SURFACE_PARMS={'delta1':0.,'delta2':0.}#correction factor in surface unit cell
inst = model.Instrument(wavel = wal, alpha = 2.0)
bulk = model.Slab(T_factor='B')
surface =  model.Slab(c = 1.0,T_factor='B')
rgh=UserVars()
rgh.new_var('beta', 0.0)#roughness factor
rgh.new_var('mu',1)#liquid film thickness
scales=['scale_CTR']
for scale in scales:
    rgh.new_var(scale,1.)

#set up experimental constant#
re = 2.818e-5#electron radius
kvect=2*np.pi/wal#k vector
Egam = 6.626*(10**-34)*3*(10**8)/wal*10**10/1.602*10**19#energy in ev
LAM=1.5233e-22*Egam**6 - 1.2061e-17*Egam**5 + 2.5484e-13*Egam**4 + 1.6593e-10*Egam**3 + 1.9332e-06*Egam**2 + 1.1043e-02*Egam
exp_const = 4*kvect/LAM
auc=unitcell.a*unitcell.b*np.sin(unitcell.gamma)

##############################################end of main setup zone############################################
#                                                                                                              #
#                                                                                                              #
#                           You seldomly need to touch script lines hereafter!!!                               #
#                                                                                                              #
#                                                                                                              #
#                                                                                                              #
################################################################################################################
#depository path for output files(structure model files(.xyz,.cif), optimized values (CTR,RAXR,E_Density) for plotting
output_file_path=output_path.module_path_locator()

################################################build up ref domains############################################
#add atoms for bulk and two ref domains (ref_domain1<half layer> and ref_domain2<full layer>)                  #
#In those two reference domains, the atoms are ordered according to first hight (z values), then y values      #
#it is a super surface structure by stacking the surface slab on bulk slab, the repeat vector was counted      #
################################################################################################################
batch_path_head=batch_path.module_path_locator()
domain_creator.add_atom_in_slab(bulk,os.path.join(batch_path_head,'Cu100_bulk.str'))
domain_creator.add_atom_in_slab(surface,os.path.join(batch_path_head,'Cu100_surface.str'))

###################################fitting function part##########################################
VARS=vars()#pass local variables to sim function
if COUNT_TIME:t_1=datetime.now()

def Sim(data,VARS=VARS):
    for command in commands:eval(command)
    VARS=VARS
    F =[]
    bv=0
    bv_container={}
    fom_scaler=[]
    beta=rgh.beta
    SCALES=[getattr(rgh,scale) for scale in scales]
    total_wt=0

    for i in range(DOMAIN_NUMBER):
        #grap wt for each domain and cal the total wt
        vars()['wt_domain'+str(int(i+1))]=VARS['rgh_domain'+str(int(i+1))].wt
        total_wt=total_wt+vars()['wt_domain'+str(int(i+1))]

    sample = model.Sample(inst, bulk, [surface], unitcell)

    if COUNT_TIME:t_2=datetime.now()

    #cal structure factor for each dataset in this for loop
    #fun to deal with the symmetrical shape of 10,30 and 20L rod at positive and negative sides
    def formate_hkl(h_,k_,x_):
        new_h,new_k,new_x=[],[],[]
        if np.around(h_[0],0) in [1,2,3] and np.around(k_[0],0)==0:
            for iii in range(len(x_)):
                if x_[iii]>0:
                    new_h.append(-h_[iii])
                    new_k.append(-k_[iii])
                    new_x.append(-x_[iii])
                else:
                    new_h.append(h_[iii])
                    new_k.append(k_[iii])
                    new_x.append(x_[iii])
            return  np.array(new_h),np.array(new_k),np.array(new_x)
        else:
            return np.array(h_),np.array(k_),np.array(x_)

    i=0
    for data_set in data:
        f=np.array([])
        h = data_set.extra_data['h']
        k = data_set.extra_data['k']
        x = data_set.x
        y = data_set.extra_data['Y']
        LB = data_set.extra_data['LB']
        dL = data_set.extra_data['dL']

        if data_set.use:
            if data_set.x[0]>100:#doing RAXR calculation(x is energy column typically in magnitude of 10000 ev)
                h_,k_,y_=formate_hkl(h,k,y)
                rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(y-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
                f=sample.cal_structure_factor_hematite_RAXR(i,VARS,RAXR_FIT_MODE,RESONANT_EL_LIST,RAXR_EL,h_, k_, y_, x, E0, F1F2,SCALES,rough)
                F.append(abs(f))
                fom_scaler.append(1)
                i+=1
            else:#doing CTR calculation (x is perpendicular momentum transfer L typically smaller than 15)
                h_,k_,x_=formate_hkl(h,k,x)
                rough = (1-beta)/((1-beta)**2 + 4*beta*np.sin(np.pi*(x-LB)/dL)**2)**0.5#roughness model, double check LB and dL values are correctly set up in data file
                if h[0]==0 and k[0]==0:#consider layered water only for specular rod if existent
                    q=np.pi*2*unitcell.abs_hkl(h_,k_,x_)
                    pre_factor=(np.exp(-exp_const*rgh.mu/q))*(4*np.pi*re/auc)*3e6
                    f = pre_factor*SCALES[0]*rough*sample.calc_f(h_, k_, x_)
                else:
                    f = rough*sample.calc_f(h_, k_, x_)
                F.append(abs(f))
                fom_scaler.append(1)
        else:
            if x[0]>100:
                i+=1
            f=np.zeros(len(y))
            F.append(f)
            fom_scaler.append(1)

    #some ducumentation about using this script#
    print_help_info=False
    if print_help_info:
        setup_domain_hematite_rcut.print_help_doc()
    #do this in shell 'model.script_module.setup_domain_hematite_rcut.print_help_doc()' to get help info

    #output how fast the code is running#
    if COUNT_TIME:t_3=datetime.now()
    if COUNT_TIME:
        print "It took "+str(t_1-t_0)+" seconds to setup"
        print "It took "+str(t_2-t_1)+" seconds to calculate bv weighting"
        print "It took "+str(t_3-t_2)+" seconds to calculate structure factor"

    #you may play with the weighting rule by setting eg 2**bv, 5**bv for the wt factor, that way you are pushing the GenX to find a fit btween a good fit (low wt factor) and a reasonable fit (high wt factor)
    return F,1+WT_BV*bv,fom_scaler
