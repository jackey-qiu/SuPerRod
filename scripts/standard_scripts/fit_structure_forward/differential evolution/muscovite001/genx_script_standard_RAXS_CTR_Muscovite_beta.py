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
import models.domain_creator_sorbate as domain_creator_sorbate
import supportive_functions.make_parameter_table_GenX_5_beta as make_grid
import convert_files

##==========================================<program begins from here>=========================================##
COUNT_TIME=False
if COUNT_TIME:t_0=datetime.now()
VERSION=1.2#version number to make easier code update to compatible with gx files based on old version scripts

##<global handles>##
RUN=1
BATCH_PATH_HEAD,OUTPUT_FILE_PATH=batch_path.module_path_locator(),output_path.module_path_locator()
F1F2=np.loadtxt(os.path.join(BATCH_PATH_HEAD,'Zr_K_edge.f1f2'))#the energy column should NOT have duplicate values after rounding up to 0 digit. If so, cut off rows of duplicate energy!
RAXR_EL,E0,NUMBER_RAXS_SPECTRA,RAXR_FIT_MODE,FREEZE='Zr',18007,21,'MI',True#FREEZE=True will have resonant el make no influence on the non-resonant structure factor. And it will otherwise.
NUMBER_DOMAIN,COHERENCE=2,True
HEIGHT_OFFSET=-2.6685#if set to 0, the top atomic layer is at 2.6685 in fractional unit before relaxation (should not be changed)
XY_OFFSET=[0,0]#takes effect only for structural atoms (not include Gaussian atoms)
GROUP_SCHEME=[[1,0]]#means group Domain1 and Domain2 for inplane and out of plane movement, set Domain2=Domain1 in side sim func

##<setting slabs>##
wal=0.7749136#wavelength,set this number right each time (read it from nQc data file)
unitcell = model.UnitCell(5.1988, 9.0266, 20.04156, 90, 95.782, 90)#a,b,c,alpha,beta,gamma, correct c each time (use q-corrected c),c=c_projected/sin(180-beta)
inst = model.Instrument(wavel = wal, alpha = 2.0)
bulk, Domain1, Domain2 = model.Slab(T_factor='u'), model.Slab(c = 1.0,T_factor='u'), model.Slab(c = 1.0,T_factor='u')
domain_creator.add_atom_in_slab(bulk,os.path.join(BATCH_PATH_HEAD,'muscovite_001_bulk_u_corrected_new.str'),height_offset=HEIGHT_OFFSET)
domain_creator.add_atom_in_slab(Domain1,os.path.join(BATCH_PATH_HEAD,'muscovite_001_surface_Al_u_corrected_new.str'),attach='_D1',height_offset=HEIGHT_OFFSET)
domain_creator.add_atom_in_slab(Domain2,os.path.join(BATCH_PATH_HEAD,'muscovite_001_surface_Si_u_corrected_new.str'),attach='_D2',height_offset=HEIGHT_OFFSET)

##<experimental constants>##
L_max=17.34#maximum L value
sig_eff=2*np.pi/(2*np.pi/(unitcell.c*np.sin(np.pi-unitcell.beta))*L_max)/5.66#intrinsic width (due to sig) with resolution width
re = 2.818e-5#electron radius
kvect=2*np.pi/wal#k vector
Egam = 6.626*(10**-34)*3*(10**8)/wal*10**10/1.602*10**19#energy in ev
LAM=1.5233e-22*Egam**6 - 1.2061e-17*Egam**5 + 2.5484e-13*Egam**4 + 1.6593e-10*Egam**3 + 1.9332e-06*Egam**2 + 1.1043e-02*Egam
exp_const = 4*kvect/LAM
auc=unitcell.a*unitcell.b*np.sin(unitcell.gamma)

##<coordination system definition>##
x0_v,y0_v,z0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])
f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])
BASIS=np.array([unitcell.a, unitcell.b, unitcell.c])
#BASIS_SET=[[1,0,0],[0,1,0],[0.10126,0,1.0051136]]
BASIS_SET=[[1,0,0],[0,1,0],[np.tan(unitcell.beta-np.pi/2.),0,1./np.cos(unitcell.beta-np.pi/2.)]]
T=inv(np.transpose(f1(x0_v,y0_v,z0_v,*BASIS_SET)))
T_INV=inv(T)

##<Adding sorbates>##to be set##
#domain1
BUILD_GRID=3#FOR cubic structure only
LEVEL,CAP,EXTRA_SORBATE=13,[],[]
SYMMETRY,SWITCH_EXTRA_SORBATE=False,[True]*10
MIRROR_EXTRA_SORBATE=[True]*10
#NUMBER_SORBATE_LAYER,NUMBER_EL_MOTIF=1,LEVEL+len(CAP)*2+len(EXTRA_SORBATE)#1 if monomer, 2 if dimmer and so on, for square_antiprism only
NUMBER_SORBATE_LAYER=0
NUMBER_EL_MOTIF=None
if type(BUILD_GRID)==type([]):
    NUMBER_EL_MOTIF=len(BUILD_GRID)
elif type(BUILD_GRID)==int(1):
    NUMBER_EL_MOTIF=BUILD_GRID**3
INFO_LIB={'basis':BASIS,'sorbate_el':'Zr','coordinate_el':'O','T':T,'T_INV':T_INV,'oligomer_type':'polymer'}#polymer_new_rot if square_antiprism

for i in range(NUMBER_SORBATE_LAYER):
    vars()['rgh_domain1_set'+str(i+1)]=UserVars()
    geo_lib_domain1={'cent_point_offset_x':0,'cent_point_offset_y':0,'cent_point_offset_z':0,'r':2.2,'theta':59.2641329,'rot_x':0,'rot_y':0,'rot_z':0,'shift_btop':0,'shift_mid':0,'shift_cap':0,'rot_ang_attach1':0,'rot_ang_attach2':0,'rot_ang_attach3':0}
    Domain1,vars()['rgh_domain1_set'+str(i+1)]=domain_creator.add_sorbate_new(domain=Domain1,anchored_atoms=[],func=domain_creator_sorbate.OS_cubic_oligomer,geo_lib=geo_lib_domain1,info_lib=INFO_LIB,domain_tag='_D1',rgh=vars()['rgh_domain1_set'+str(i+1)],index_offset=[i*2*NUMBER_EL_MOTIF,NUMBER_EL_MOTIF+i*2*NUMBER_EL_MOTIF],xy_offset=XY_OFFSET,height_offset=HEIGHT_OFFSET,symmetry_couple=SYMMETRY,level=LEVEL,cap=CAP,attach_sorbate_number=EXTRA_SORBATE,first_or_second=SWITCH_EXTRA_SORBATE,mirror=MIRROR_EXTRA_SORBATE,build_grid=BUILD_GRID)

##<Adding Gaussian peaks>##
NUMBER_GAUSSIAN_PEAK, EL_GAUSSIAN_PEAK, FIRST_PEAK_HEIGHT=6,['O','O','O','O','O','O'],1
GAUSSIAN_OCC_INIT, GAUSSIAN_LAYER_SPACING, GAUSSIAN_U_INIT=1,2,0.1
GAUSSIAN_SHAPE, GAUSSIAN_RMS='Flat',2
Domain1, Gaussian_groups,Gaussian_group_names=domain_creator.add_gaussian(domain=Domain1,el=EL_GAUSSIAN_PEAK,number=NUMBER_GAUSSIAN_PEAK,first_peak_height=FIRST_PEAK_HEIGHT,spacing=GAUSSIAN_LAYER_SPACING,u_init=GAUSSIAN_U_INIT,occ_init=GAUSSIAN_OCC_INIT,height_offset=HEIGHT_OFFSET,c=unitcell.c,domain_tag='_D1',shape=GAUSSIAN_SHAPE,gaussian_rms=GAUSSIAN_RMS)
for i in range(len(Gaussian_groups)):vars()[Gaussian_group_names[i]]=Gaussian_groups[i]
rgh_gaussian=domain_creator.define_gaussian_vars(rgh=UserVars(),domain=Domain1,shape=GAUSSIAN_SHAPE)

'''WARNING! Choose one way to freeze element. Errors will appear if using both ways.'''
##<Freeze Elements by specifing values>##
U_RAXS_LIST=[]
OC_RAXS_LIST=[]
X_RAXS_LIST=[]
Y_RAXS_LIST=[]
Z_RAXS_LIST=np.array([])/unitcell.c - 1.
el_freezed=RAXR_EL
Domain1=domain_creator.add_freezed_els(domain=Domain1,el=el_freezed,u=U_RAXS_LIST,oc=OC_RAXS_LIST,x=X_RAXS_LIST,y=Y_RAXS_LIST,z=Z_RAXS_LIST)

##<Freeze Elements using adding_gaussian function>##
NUMBER_GAUSSIAN_PEAK_FREEZE, EL_GAUSSIAN_PEAK_FREEZE, FIRST_PEAK_HEIGHT_FREEZE=6,RAXR_EL,5
GAUSSIAN_OCC_INIT_FREEZE, GAUSSIAN_LAYER_SPACING_FREEZE, GAUSSIAN_U_INIT_FREEZE=1,2,0.1
GAUSSIAN_SHAPE_FREEZE, GAUSSIAN_RMS_FREEZE='Flat',2
Domain1, Gaussian_groups_freeze,Gaussian_group_names_freeze=domain_creator.add_gaussian(domain=Domain1,el=EL_GAUSSIAN_PEAK_FREEZE,number=NUMBER_GAUSSIAN_PEAK_FREEZE,first_peak_height=FIRST_PEAK_HEIGHT_FREEZE,spacing=GAUSSIAN_LAYER_SPACING_FREEZE,u_init=GAUSSIAN_U_INIT_FREEZE,occ_init=GAUSSIAN_OCC_INIT_FREEZE,height_offset=HEIGHT_OFFSET,c=unitcell.c,domain_tag='_D1',shape=GAUSSIAN_SHAPE_FREEZE,gaussian_rms=GAUSSIAN_RMS_FREEZE,freeze_tag=True)
for i in range(len(Gaussian_groups_freeze)):vars()[Gaussian_group_names_freeze[i]]=Gaussian_groups_freeze[i]
rgh_gaussian_freeze=domain_creator.define_gaussian_vars(rgh=UserVars(),domain=Domain1,shape=GAUSSIAN_SHAPE_FREEZE)

##<Define atom groups>##
#surface atoms
#old stuff (to be deleted)
group_number=1##to be set##(number of groups to be considered for model fit)
groups,group_names,atom_group_info=domain_creator.setup_atom_group_muscovite(domain=[Domain1,Domain2],group_number=group_number)
for i in range(len(groups)):vars()[group_names[i]]=groups[i]
#group atom layers using methosd from Sang Soo Lee Matlab script
names,layer_groups=domain_creator.setup_atom_group_muscovite_2(domain=[Domain1,Domain2])
for i in range(len(layer_groups)):vars()[names[i]]=layer_groups[i]

#sorbate_atoms
sorbate_id_list_domain1,sorbate_group_names_domain1=domain_creator.generate_sorbate_ids(Domain1,NUMBER_SORBATE_LAYER,INFO_LIB['sorbate_el'],NUMBER_EL_MOTIF,symmetry=SYMMETRY,level=CAP)
sorbate_atom_group_info=[{'domain':Domain1,'ref_id_list':sorbate_id_list_domain1,'ref_group_names':sorbate_group_names_domain1,'ref_sym_list':[],'domain_tag':''}]
sorbate_groups,sorbate_group_names=domain_creator.setup_atom_group(gp_info=sorbate_atom_group_info)
for i in range(len(sorbate_groups)):vars()[sorbate_group_names[i]]=sorbate_groups[i]

##<Define other pars>##
rgh=domain_creator.define_global_vars(rgh=UserVars(),domain_number=NUMBER_DOMAIN)#global vars
rgh_raxs=domain_creator.define_raxs_vars(rgh=UserVars(),number_spectra=NUMBER_RAXS_SPECTRA,number_domain=1)#RAXR spectra pars
rgh_dlw=domain_creator.define_diffused_layer_water_vars(rgh=UserVars())#Diffused Layered water pars
rgh_dls=domain_creator.define_diffused_layer_sorbate_vars(rgh=UserVars())#Diffused Layered sorbate pars

##<make fit table file>##
if not RUN:
    table_container=[]
    rgh_instance_list=[rgh]+groups+sorbate_groups+Gaussian_groups+[rgh_gaussian]+[rgh_gaussian_freeze]+[vars()['rgh_domain1_set'+str(i+1)] for i in range(NUMBER_SORBATE_LAYER)]+[rgh_dlw,rgh_dls]
    rgh_instance_name_list=['rgh']+group_names+sorbate_group_names+Gaussian_group_names+['rgh_gaussian']+['rgh_gaussian_freeze']+['rgh_domain1_set'+str(i+1) for i in range(NUMBER_SORBATE_LAYER)]+['rgh_dlw','rgh_dls']
    table_container=make_grid.set_table_input_all(container=table_container,rgh_instance_list=rgh_instance_list,rgh_instance_name_list=rgh_instance_name_list,par_file=os.path.join(BATCH_PATH_HEAD,'pars_ranges.txt'))
    #raxs pars
    table_container=make_grid.set_table_input_raxs(container=table_container,rgh_group_instance=rgh_raxs,rgh_group_instance_name='rgh_raxs',par_range={'a':[0,20],'b':[-5,5],'c':[0,10],'A':[0,10],'P':[0,1]},number_spectra=NUMBER_RAXS_SPECTRA,number_domain=1)
    #build up the tab file
    make_grid.make_table(container=table_container,file_path=os.path.join(OUTPUT_FILE_PATH,'par_table.tab'))

##<fitting function part>##
if COUNT_TIME:t_1=datetime.now()
VARS=vars()
def Sim(data,VARS=VARS):

    ##<update the basis info>##
    INFO_LIB['basis']=np.array([unitcell.a, unitcell.b, unitcell.c])
    ##<Extract pars>##
    layered_water_pars=vars(rgh_dlw)
    layered_sorbate_pars=vars(rgh_dls)
    raxs_vars=vars(rgh_raxs)

    ##<update sorbates>##
    [domain_creator.update_sorbate_new(domain=Domain1,anchored_atoms=[],func=domain_creator_sorbate.OS_cubic_oligomer,info_lib=INFO_LIB,domain_tag='_D1',rgh=VARS['rgh_domain1_set'+str(i+1)],index_offset=[i*2*NUMBER_EL_MOTIF,NUMBER_EL_MOTIF+i*2*NUMBER_EL_MOTIF],xy_offset=XY_OFFSET,height_offset=HEIGHT_OFFSET,level=LEVEL,symmetry_couple=SYMMETRY,cap=CAP,attach_sorbate_number=EXTRA_SORBATE,first_or_second=SWITCH_EXTRA_SORBATE,mirror=MIRROR_EXTRA_SORBATE,build_grid=BUILD_GRID) for i in range(NUMBER_SORBATE_LAYER)]#domain1

    ##<update gaussian peaks>##
    if NUMBER_GAUSSIAN_PEAK>0:
        domain_creator.update_gaussian(domain=Domain1,rgh=rgh_gaussian,groups=Gaussian_groups,el=EL_GAUSSIAN_PEAK,number=NUMBER_GAUSSIAN_PEAK,height_offset=HEIGHT_OFFSET,c=unitcell.c,domain_tag='_D1',shape=GAUSSIAN_SHAPE,print_items=False,use_cumsum=True)
    if NUMBER_GAUSSIAN_PEAK_FREEZE>0:
        domain_creator.update_gaussian(domain=Domain1,rgh=rgh_gaussian_freeze,groups=Gaussian_groups_freeze,el=EL_GAUSSIAN_PEAK_FREEZE,number=NUMBER_GAUSSIAN_PEAK_FREEZE,height_offset=HEIGHT_OFFSET,c=unitcell.c,domain_tag='_D1',shape=GAUSSIAN_SHAPE_FREEZE,print_items=False,use_cumsum=True,freeze_tag=True)

    ##<link groups>##
    #[eval(each_command) for each_command in domain_creator.link_atom_group(gp_info=atom_group_info,gp_scheme=GROUP_SCHEME)]
    domain_creator.setup_atom_group_2(VARS)

    ##<format domains>##
    domain={'domains':[Domain1,Domain2],'layered_water_pars':layered_water_pars,'layered_sorbate_pars':layered_sorbate_pars,\
            'global_vars':rgh,'raxs_vars':raxs_vars,'F1F2':F1F2,'E0':E0,'el':RAXR_EL,'freeze':FREEZE,'exp_factors':[exp_const,rgh.mu,re,auc,rgh.ra_conc],'sig_eff':sig_eff}
    sample = model.Sample(inst, bulk, domain, unitcell,coherence=COHERENCE,surface_parms={'delta1':0.,'delta2':0.})

    ##<calculate structure factor>##
    F,fom_scaler=[],[]
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
            if x[0]>100:
                i+=1
                q=np.pi*2*unitcell.abs_hkl(h,k,y)
                rough = (1-rgh.beta)**2/(1+rgh.beta**2 - 2*rgh.beta*np.cos(q*unitcell.c*np.sin(np.pi-unitcell.beta)/2))
                pre_factor=np.exp(-exp_const*rgh.mu/q)*(4*np.pi*re/auc)**2/q**2
            else:
                q=np.pi*2*unitcell.abs_hkl(h,k,x)
                rough = (1-rgh.beta)**2/(1+rgh.beta**2 - 2*rgh.beta*np.cos(q*unitcell.c*np.sin(np.pi-unitcell.beta)/2))
                pre_factor=np.exp(-exp_const*rgh.mu/q)*(4*np.pi*re/auc)**2/q**2
            f=abs(sample.calculate_structure_factor(h,k,x,y,index=i,fit_mode=RAXR_FIT_MODE,height_offset=HEIGHT_OFFSET*unitcell.c,version=VERSION))
            F.append(3e6*pre_factor*rough*f*f)
            fom_scaler.append(1)
        else:
            if x[0]>100:
                i+=1
            f=np.zeros(len(y))
            F.append(f)
            fom_scaler.append(1)
    if COUNT_TIME:
        t_2=datetime.now()

    ##<print structure/plotting files>##
    if not RUN:
        domain_creator.print_structure_files_muscovite_new(domain_list=[Domain1,Domain2],z_shift=0.8+HEIGHT_OFFSET,number_gaussian=NUMBER_GAUSSIAN_PEAK,el=RAXR_EL,matrix_info=INFO_LIB,save_file=OUTPUT_FILE_PATH)
        create_plots.generate_plot_files(output_file_path=OUTPUT_FILE_PATH,sample=sample,rgh=rgh,data=data,fit_mode=RAXR_FIT_MODE,z_min=-5,z_max=50,RAXR_HKL=[0,0,20],height_offset=HEIGHT_OFFSET*BASIS[2],version=VERSION,freeze=FREEZE)
        #make sure the tab_file is saved in the dumped files directory before running this function
        #domain_creator.print_data_for_publication_B3_muscovite(N_sorbate=NUMBER_GAUSSIAN_PEAK+len(U_RAXS_LIST)+NUMBER_GAUSSIAN_PEAK_FREEZE,domain=Domain1,z_shift=0.8+HEIGHT_OFFSET+0.8666,save_file=os.path.join(OUTPUT_FILE_PATH,'temp_publication_data_muscovite.xyz'),tab_file=os.path.join(OUTPUT_FILE_PATH,'best_fit_pars.tab'))

        #then do this command inside shell to extract the errors for A and P: model.script_module.create_plots.append_errors_for_A_P(par_instance=model.parameters,dump_file=os.path.join(model.script_module.OUTPUT_FILE_PATH,'temp_plot_raxr_A_P_Q'),raxs_rgh='rgh_raxs')
        make_dummy_data,combine_data_sets=False,False
        if make_dummy_data:
            domain_creator.make_dummy_data(file=os.path.join(OUTPUT_FILE_PATH,'temp_dummy_data.dat'),data=data,I=F)
        if combine_data_sets:
            domain_creator.combine_all_datasets(file=os.path.join(OUTPUT_FILE_PATH,'temp_full_dataset.dat'),data=data)
        atm_number={'Zr':40}
        convert_files.convert_best_pars_to_matlab_input_file(file_name=os.path.join(OUTPUT_FILE_PATH,'temp_matlab_param.dat'),domain=Domain1,layered_water=rgh_dlw,c=unitcell.c,rgh=rgh,scale=inst.get_inten(),vars=VARS,freeze=FREEZE,z_raxr=atm_number[RAXR_EL])
    if COUNT_TIME:
        t_3=datetime.now()
        print "It took "+str(t_1-t_0)+" seconds to setup"
        print "It took "+str(t_2-t_1)+" seconds to do calculation for one generation"
        print "It took "+str(t_3-t_2)+" seconds to generate output files"
    return F,1,fom_scaler
    ##========================================<program ends here>========================================================##
