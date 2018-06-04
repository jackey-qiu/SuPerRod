import numpy as np
#estimate errors of bond length based on errors of geometrical parameters
edge=2.78
alpha,alpha_error=73.2,0.1
theta,theta_error=-6.8,0.1
delta1,delta1_error=0.087,0.003
delta2,delta2_error=-0.1,0.01
a_l,a_r=alpha-alpha_error,alpha+alpha_error
print 'error of PbO1 ',edge/4.*(np.reciprocal(np.sin(np.deg2rad(a_l/2)))-np.reciprocal(np.sin(np.deg2rad(a_r/2))))+delta1_error
print 'error of PbO2 ',edge/4.*(np.reciprocal(np.sin(np.deg2rad(a_l/2)))-np.reciprocal(np.sin(np.deg2rad(a_r/2))))
print 'error of PbOdistal ',edge/4.*(np.reciprocal(np.sin(np.deg2rad(a_l/2)))-np.reciprocal(np.sin(np.deg2rad(a_r/2))))+delta2_error
print 'error of O1PbO2 ',alpha_error
print 'error of O1PbOdistal ',alpha_error+theta_error
print 'error of Pb-Fe distance ', edge/4.*(np.reciprocal(np.tan(np.deg2rad(a_l/2)))-np.reciprocal(np.tan(np.deg2rad(a_r/2))))
    