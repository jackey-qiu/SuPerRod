import numpy as np
import os

def convert_best_pars_to_matlab_input_file(file_name=None,domain=None,layered_water=None,c=20.04156,rgh=None,scale=1,vars=None):
    data=domain._extract_values()
    z=np.array(data[2])*c*np.sin((180-95.787)/180.*np.pi)
    c_projected=c*np.sin((180-95.787)/180.*np.pi)
    #z=np.array(data[2])*20.1058*np.sin((180-90)/180.*np.pi)
    z_new=list(z-z[0])#first atom is mica surface, now mica surface has a z value of 0
    f=open(file_name,'w')
    f.write('% Param No. Best-Fit     Opt(1,2)             FIT    Initial      Std Dev\n\n')
    index_gaussian_peaks=[np.where(domain.id==each_id)[0][0] for each_id in domain.id if 'Gaussian_' in each_id]
    index_freeze_peaks=[np.where(domain.id==each_id)[0][0] for each_id in domain.id if 'Freezed_' in each_id]
    z_freeze=[z_new[each_index] for each_index in index_freeze_peaks]
    z_gaussian=[z_new[each_index] for each_index in index_gaussian_peaks]
    #for each_index in index_gaussian_peaks:
    u_gaussian=[data[4][each_index] for each_index in index_gaussian_peaks]
    u_freeze=[data[4][each_index] for each_index in index_freeze_peaks]
    oc_gaussian=[data[5][each_index]/4.0/2.0 for each_index in index_gaussian_peaks]#occ normalized to the half surface unit cell
    oc_freeze=[data[5][each_index]/4.0/2.0 for each_index in index_freeze_peaks]
    ref_height_offset=0
    try:
        beta=rgh.beta+1
        MU=rgh.mu
    except:
        beta=1.05
        MU=10
    if vars!=None:
        relaxation_top=[vars['gp_K_layer_1'].getdz(),vars['gp_Obas_layer_1'].getdz(),vars['gp_Octa_layer_1'].getdz(),vars['gp_Octt_layer_1'].getdz(),vars['gp_Otop_layer_1'].getdz()]
        relaxation_mid=[vars['gp_K_layer_2'].getdz(),vars['gp_Obas_layer_2'].getdz(),vars['gp_Octa_layer_2'].getdz(),vars['gp_Octt_layer_1'].getdz(),vars['gp_Otop_layer_2'].getdz()]
        relaxation_bot=[vars['gp_K_layer_3'].getdz(),vars['gp_Obas_layer_3'].getdz(),vars['gp_Octa_layer_3'].getdz(),vars['gp_Octt_layer_3'].getdz(),vars['gp_Otop_layer_3'].getdz()]
        ref_height_offset=-vars['gp_Otop_layer_1'].getdz()*c_projected
    else:
        relaxation_top=[0]*5
        relaxation_mid=[0]*5
        relaxation_bot=[0]*5
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(1   ,    scale  ,   0.0010  ,   1.0000  ,   1   ,  scale   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(2   ,    beta  ,   0.0010  ,   0.1  ,   0   ,  beta   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(3   ,    MU  ,   0.0010  ,   5  ,   0   ,  MU   ,  0.00))
    #water layered
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(4   ,    layered_water.getFirst_layer_height_w()+1+ref_height_offset  ,   0.0010  ,   1.0000  ,   1   ,  layered_water.getFirst_layer_height_w()+1   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(5   ,    layered_water.getU0_w()+1  ,   0.0010  ,   1.0000  ,   1   ,  layered_water.getU0_w()+1   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(6   ,    layered_water.getD_w()+1  ,   0.0010  ,   1.0000  ,   1   ,  layered_water.getD_w()+1   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(7   ,    layered_water.getUbar_w()+1  ,   0.0010  ,   1.0000  ,   1   ,  layered_water.getUbar_w()+1   ,  0.00))
    #adsorbed water
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(8   ,    1  ,   0.0010  ,   0.5  ,   0   ,  1   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(9   ,    1  ,   0.0010  ,   0.5  ,   0   ,  1   ,  0.00))
    f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(10   ,    1  ,   0.0010  ,   0.5  ,   0   ,  1   ,  0.00))

    for i in range(3):
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(11+i*3   ,    z_gaussian[i]+1+ref_height_offset  ,   0.0010  ,   0.5  ,   1   ,  z_gaussian[i]+1+ref_height_offset   ,  0.00))
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(12+i*3   ,    oc_gaussian[i]+1  ,   0.0010  ,   0.5  ,   1   ,  oc_gaussian[i]+1   ,  0.00))
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(13+i*3   ,    u_gaussian[i]+1  ,   0.0010  ,   0.5  ,   1   ,  u_gaussian[i]+1   ,  0.00))
    for i in range(11):
        if 20+i in [20,21,22,23,24,25]:
            f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(20+i   ,    1  ,   0.0010  ,   0.5  ,   0   ,  1   ,  0.00))
        else:
            f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(20+i   ,    1  ,   0.0010  ,   0.5  ,   1   ,  1   ,  0.00))
    for i in range(5):
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(31+i   ,    1+relaxation_bot[i]*c_projected  ,   0.0010  ,   0.5  ,   1   ,  1+relaxation_bot[i]*c_projected   ,  0.00))
    for i in range(5):
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(36+i   ,    1+relaxation_mid[i]*c_projected  ,   0.0010  ,   0.5  ,   1   ,  1+relaxation_mid[i]*c_projected   ,  0.00))
    for i in range(5):
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(41+i   ,    1+relaxation_top[i]*c_projected  ,   0.0010  ,   0.5  ,   1   ,  1+relaxation_top[i]*c_projected   ,  0.00))
    for i in range(len(z_gaussian)-3):
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(46+i*3   ,    z_gaussian[i+3]+1+ref_height_offset  ,   0.0010  ,   0.5  ,   1   ,  z_gaussian[i+3]+1+ref_height_offset   ,  0.00))
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(47+i*3   ,    oc_gaussian[i+3]+1  ,   0.0010  ,   0.5  ,   1   ,  oc_gaussian[i+3]+1   ,  0.00))
        f.write('%i    %10.6f    %6.4f    %6.4f    %i    %10.7f    %10.6f\n'%(48+i*3   ,    u_gaussian[i+3]+1  ,   0.0010  ,   0.5  ,   1   ,  u_gaussian[i+3]+1   ,  0.00))
    #now write the freeze RAXR element
    f.write('occ_ra=['+';'.join(map(lambda i:str(i),oc_freeze))+'];\n')
    f.write('pos_ra=['+';'.join(map(lambda i:str(i),np.array(z_freeze)+ref_height_offset))+'];\n')
    f.write('u_ra=['+';'.join(map(lambda i:str(i),u_freeze))+'];\n')
    f.close()
