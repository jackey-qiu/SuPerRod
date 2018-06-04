import os
import numpy as np
from numpy.linalg import inv
from datetime import datetime
import models.sxrd_new1 as model
from models.utils import UserVars
import batchfile.locate_path as batch_path
import dump_files.locate_path as output_path
import models.domain_creator as domain_creator
import supportive_functions.create_plots as create_plots
import pickle

##==========================================<program begins from here>=========================================##
run=0
N_el=40
h,k,l,zs=[],[],[],[]
water_scaling=None

if not run:#do this once to extract all necessary information
    PATH=output_path.module_path_locator()
    e_data_file=os.path.join(PATH,"temp_plot_eden_fourier_synthesis")
    e_data=np.append([pickle.load(open(e_data_file,"rb"))[0]],[pickle.load(open(e_data_file,"rb"))[1]],axis=0).transpose()
    ctr_data=np.loadtxt(os.path.join(PATH,"temp_full_dataset.dat"))
    ##formate e density data
    e_data_formate=np.append(e_data[:,0][:,np.newaxis],np.zeros((len(e_data),3)),axis=1)
    e_data_formate=np.append(e_data_formate,e_data[:,1][:,np.newaxis],axis=1)
    e_data_formate=np.append(e_data_formate,np.zeros((len(e_data),3))+0.0001,axis=1)
    np.savetxt(os.path.join(PATH,"e_data_temp.dat"),e_data_formate)
    ##formate AP values
    AP_Q_file=os.path.join(PATH,"temp_plot_raxr_A_P_Q")
    data_AP_Q=pickle.load(open(AP_Q_file,"rb"))
    AP_data_formate_A=np.append(np.array(data_AP_Q[1][2])[:,np.newaxis],np.zeros((len(data_AP_Q[1][2]),3)),axis=1)
    AP_data_formate_A=np.append(AP_data_formate_A,np.array(data_AP_Q[1][0])[:,np.newaxis],axis=1)
    AP_data_formate_A=np.append(AP_data_formate_A,np.max(data_AP_Q[1][3],axis=1)[:,np.newaxis],axis=1)
    AP_data_formate_A=np.append(AP_data_formate_A,np.zeros((len(data_AP_Q[1][2]),2))+0.0001,axis=1)

    AP_data_formate_P=np.append(np.array(data_AP_Q[1][2])[:,np.newaxis]+3.5,np.zeros((len(data_AP_Q[1][2]),3)),axis=1)#here 3.5 is a arbitrary q offset, make sure it is larger than the maxium of q vlues
    AP_data_formate_P=np.append(AP_data_formate_P,np.array(data_AP_Q[1][1])[:,np.newaxis],axis=1)
    AP_data_formate_P=np.append(AP_data_formate_P,np.max(data_AP_Q[1][4],axis=1)[:,np.newaxis],axis=1)
    AP_data_formate_P=np.append(AP_data_formate_P,np.zeros((len(data_AP_Q[1][2]),2))+0.0001,axis=1)

    AP_data_formate=np.append(AP_data_formate_A,AP_data_formate_P,axis=0)
    np.savetxt(os.path.join(PATH,"APQ_data_temp.dat"),AP_data_formate)

    ##extract hkl from ctr_data
    h,k,l=[],[],[]
    for i in range(len(ctr_data)):
        if ctr_data[i,3]!=0:
            if ctr_data[i,3] not in l:
                h.append(ctr_data[i,1])
                k.append(ctr_data[i,2])
                l.append(ctr_data[i,3])
    print 'h=np.array([',','.join(map(lambda x:str(x),h)),'])'
    print 'k=np.array([',','.join(map(lambda x:str(x),k)),'])'
    print 'l=np.array([',','.join(map(lambda x:str(x),l)),'])'


    ##find z of peaks
    fit_range=[0,40]
    zs=[]
    zs_sensor=None
    if zs_sensor==None:
        for i in range(1,len(e_data)-1):
            if e_data[i-1,1]<e_data[i,1] and e_data[i+1,1]<e_data[i,1]:
                zs.append(e_data[i,0])
    elif type(zs_sensor)==int:
        zs=[fit_range[0]+(fit_range[1]-fit_range[0])/zs_sensor*i for i in range(zs_sensor)]+[fit_range[1]]
    else:
        zs=np.array(zs_sensor)
    print 'zs=np.array([',','.join(map(lambda x:str(x),zs)),'])'

    ##extract water scaling value##
    water_scaling_file=os.path.join(PATH,"water_scaling")
    water_scaling=pickle.load(open(water_scaling_file,"rb"))[-1]
    print 'water_scaling=',water_scaling

else:
    h=np.array([ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ])
    k=np.array([ 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0 ])
    l=np.array([ 0.41,0.53,0.61,0.75,0.88,1.15,1.45,1.71,2.31,2.64,2.85,3.21,3.55,4.24,4.55,5.61,6.25,7.31,9.15,10.31,11.15 ])
    zs=np.array([ 1.5,3.65,5.85,7.7,9.7,11.75,17.05,19.1,21.1,23.1,25.25,27.45,29.55,32.0,34.45,36.5,38.6,40.65,42.55,44.65,48.0,49.85 ])
    water_scaling= 0.33

##cal q list
q_list=np.array(create_plots.q_list_func(h,k,l))

##define fitting parameters
rgh_u=UserVars()
rgh_oc=UserVars()
rgh_dz=UserVars()
for i in range(len(zs)):
    rgh_u.new_var('u'+str(i+1),0.2)
    rgh_oc.new_var('oc'+str(i+1),0.)
    rgh_dz.new_var('dz'+str(i+1),0.)

def Sim(data):
    u_list=[]
    oc_list=[]
    z_list=[]
    ##<Extract pars>##
    for i in range(len(zs)):
        u_list.append(getattr(rgh_u,'getU'+str(0+1))())
        oc_list.append(getattr(rgh_oc,'getOc'+str(i+1))())
        z_list.append(getattr(rgh_dz,'getDz'+str(i+1))()+zs[i])

    ##<calculate e density>##
    F,fom_scaler=[],[]
    i=0
    for data_set in data:
        if data_set.x[-1]>10:
            f=np.array([])
            z = data_set.x
            A,P,Q=create_plots.find_A_P_muscovite(q_list,z_list,oc_list,u_list)
            f=np.array(create_plots.fourier_synthesis(q_list,P,A,z,N_el))/water_scaling
            #f=create_plots.cal_e_density(z_list,oc_list,u_list,z_min=z[0],z_max=z[-1],water_scaling=water_scaling)

            F.append(f)
            fom_scaler.append(1)
        else:
            f=np.array([])
            q = data_set.x
            A,P,Q=create_plots.find_A_P_muscovite(q_list,z_list,oc_list,u_list)
            F.append(np.append(A,P))
            fom_scaler.append(1)

    print_items=0
    if print_items:
        print 'U_RAXS_LIST=['+','.join(map(lambda u:str(u**2),u_list))+']'
        print 'OC_RAXS_LIST=['+','.join(map(lambda oc:str(oc),oc_list))+']'
        print 'Z_RAXS_LIST=['+','.join(map(lambda z:str(z),z_list))+']'
        print 'X_RAXS_LIST=[0]*'+str(len(u_list))
        print 'Y_RAXS_LIST=[0]*'+str(len(u_list))

    return F,1,fom_scaler
    ##========================================<program ends here>========================================================##
