#!/usr/bin/python
from mpi4py import MPI
import numpy as np
from numpy import *
from datetime import datetime
genxpath = '/home/qiu05/genx_pc_qiu'
#genxpath=os.path.abspath('..')
import sys
import time
sys.path.insert(0,genxpath)
#sys.path.append(genxpath+'/geometry_modules')
import model, time, fom_funcs
import diffev_mpi as diffev
import filehandling as io
##new in version 2##
#errro bar for each par will be calculated before program being halted
##new in version 3##
#find the best control parameters (pop size:N, mutation constant:km, cross_over constant: kr, mutation propability:pf) and trial methods for fitting

comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

def find_boundary(n_process,n_jobs,rank):
    step_len=int(n_jobs/n_process)
    remainder=int(n_jobs%n_process)
    left,right=0,0
    if rank<=remainder-1:
        left=rank*(step_len+1)
        right=(rank+1)*(step_len+1)-1
    elif rank>remainder-1:
        left=remainder*(step_len+1)+(rank-remainder)*step_len
        right=remainder*(step_len+1)+(rank-remainder+1)*step_len-1
    return left,right
# Okay lets make it possible to batch script this file ...
if len(sys.argv) !=3:
    print sys.argv
    print 'Wrong number of arguments to %s'%sys.argv[0]
    print 'Usage: %s infile.gx'%sys.argv[0]
    sys.exit(1)
   
infile = sys.argv[1]
pop_num=int(sys.argv[2])

t_start_0=datetime.now()
###############################################################################
# Parameter section - modify values according to your needs
###############################################################################
#
# To leave any of the control parameters unchanged with respect to the ones in
# the loaded .gx file, you need to comment out the corresponding variables
# below

# List of repetition numbers.
# For each number in the list, one run will be performed for each distinct com-
# bination of km, kr, and fom  parameters (see below). The run files will be
# named according to these numbers
# e.g. range(5)    --> [0,1,2,3,4] (total of 5 repetitions named 0-4)
#      range(5,10) --> [5,6,7,8,9] (total of 5 repetitions starting with 5)
#      [1]         --> [1] (one iteration with index 1)
#iter_list = range(5)

#print "the run starts @ %s"%(str(datetime.now()))
iter_list = [1]
#####################
# figure of merit (FOM) to use
# needs to be a list of strings, valid names are:
#   'R1'
#   'R2'
#   'log'
#   'diff'
#   'sqrt'
#   'chi2bars'
#   'chibars'
#   'logbars'
#   'sintth4'
# e.g.: fom_list = ['log','R1']  # performs all repetitions for 'log' and 'R1'
fom_list = ['chi2bars']

# diffev control parameters
# needs to be a list of parameters combinations to use. 
# example:
#   krkm_list = [[0.7,0.8], [0.9,0.95]]
#   will run fits with these parameter combinations:
#   1. km = 0.7, kr = 0.8
#   2. km = 0.9, kr = 0.95
#krkm_list = [[0.1,0.9],[0.3,0.9],[0.5,0.9],[0.7,0.9],[0.9,0.9]]
krkmPf_list =[[0.9,0.9,0.8]]

# NOT YET WORKING!!!
create_trial = ['best_1_bin']#'best_1_bin','rand_1_bin',#'best_either_or','rand_either_or'

# Population size
use_pop_mult = False             # absolute (F) or relative (T) population size
pop_mult = 8 			 # if use_pop_mult = True, populatio multiplier
pop_size = pop_num        # if use_pop_mult = False, population size

# Generations
use_max_generations = True       # absolute (T) or relative (F) maximum gen.
max_generations=pop_num*20      # if use_max_generations = True
max_generation_mult = 6          # if use_max_generations = False

# Parallel processing
use_parallel_processing = True

# Fitting
use_start_guess = False
use_boundaries = True
use_autosave = True
autosave_interval = 200
max_log = 600000

# Sleep time between generations
sleep_time = 0.000001

# Genx directory to add to the system path:
#genxpath = '/afs/umich.edu/user/c/s/cschlep/software/python/genx'


###############################################################################
# End of parameter section
#-------------------------
# DO NOT MODIFY CODE BELOW
###############################################################################
mod = model.Model()
config = io.Config()
opt = diffev.DiffEv()
#if you want create log file, uncomment the following lines
"""
logfile = '%s_%s_runlog.txt' % \
        (time.strftime('%Y%m%d_%H%m%S'), infile.replace('.gx',''))
fid = open(logfile,'w')
fid.write('# GenX logfile for %s\n' % infile)
fid.write('# %s\n' % time.asctime())
fid.write('# \n')
fid.write('# %12s %4s %4s %4s %12s\n' % ('FOM-Function', 'km', 'kr', 'iter', 'FOM'))
fid.close
"""

def autosave():
    #print 'Updating the parameters'
    mod.parameters.set_value_pars(opt.best_vec)
    io.save_gx(outfile, mod, opt, config)
    
opt.set_autosave_func(autosave)

par_list = [(trial,f,rm,i) for trial in create_trial for f in fom_list for rm in krkmPf_list \
            for i in iter_list]

tmp_fom=[]
tmp_trial_vec=[]
tmp_pop_vec=[]
tmp_fom_vec=[]

for pars in par_list:
    trial=pars[0]
    fom = pars[1]
    km = pars[2][0]  # km    
    kr = pars[2][1]  # kr
    pf=pars[2][2]
    iter = pars[3]
    
    # Load the model ...
    if rank==0: print 'Loading model %s...'%infile
    io.load_gx(infile, mod, opt, config)
    
    # Simulate, this will also compile the model script
    if rank==0: print 'Simulating model...'
    mod.simulate()
    # Setting up the solver
    eval('mod.set_fom_func(fom_funcs.%s)' % fom)
    
    # Lets set the solver parameters:
    try:
        opt.set_create_trial(trial)
    except:
        print 'Warning: create_trial is not defined in script.'   
    try:
        opt.set_kr(kr)
    except:
        print 'Warning: kr is not defined in script.'   
    try:
        opt.set_km(km)
    except :
        print 'Warning: km is not defined in script.'
    try:
        opt.pf=pf
    except:
        print 'Warning; pf is not defined in script.'
    try:
        opt.set_use_pop_mult(use_pop_mult)
    except:
        print 'Warning: use_pop_mult is not defined in script.'   
    try:
        opt.set_pop_mult(pop_mult)
    except:
        print 'Warning: pop_mult is not defined in script.'   
    try:
        opt.set_pop_size(pop_size)
    except:
        print 'Warning: pop_size is not defined in script.'   
    try:
        opt.set_use_max_generations(use_max_generations)
    except:
        print 'Warning: use_max_generations is not defined in script.'   
    try:
        opt.set_max_generations(max_generations)
    except:
        print 'Warning: max_generations is not defined in script.'   
    try:
        opt.set_max_generation_mult(max_generation_mult)
    except:
        print 'Warning: max_generation_mult is not defined in script.'   
    try:
        opt.set_use_parallel_processing(use_parallel_processing)
    except:
        print 'Warning: use_parallel_processing is not defined in script.'   
    try:
        opt.set_use_start_guess(use_start_guess)
    except:
        print 'Warning: use_start_guess is not defined in script.'   
    try:
        opt.set_use_boundaries(use_boundaries)
    except:
        print 'Warning: use_boundaries is not defined in script.'   
    try:
        opt.set_use_autosave(use_autosave)
    except:
        print 'Warning: use_autosave is not defined in script.'   
    try:
        opt.set_autosave_interval(autosave_interval)
    except:
        print 'Warning: autosave_interval is not defined in script.'   
    try:
        opt.set_max_log(max_log)
    except:
        print 'Warning: max_log is not defined in script.'   
    try:
        opt.set_sleep_time(sleep_time)
    except:
        print 'Warning: sleep_time is not defined in script.'     
    
    # Sets up the fitting ...
    if rank==0:print 'Setting up the optimizer...'
    opt.reset() # <--- Add this line
    opt.init_fitting(mod)
    #rank 0 is in charge of generating of pop vectors, and distribute to the other processors
    if rank==0:
        opt.pop_vec = [opt.par_min + np.random.rand(opt.n_dim)*(opt.par_max -\
            opt.par_min) for i in range(opt.n_pop)]
        tmp_pop_vec=opt.pop_vec
    tmp_pop_vec=comm.bcast(tmp_pop_vec,root=0)
    opt.pop_vec=tmp_pop_vec
    
    if opt.use_start_guess:
        opt.pop_vec[0] = array(opt.start_guess)
        
    opt.trial_vec = [zeros(opt.n_dim) for i in range(opt.n_pop)]
    opt.best_vec = opt.pop_vec[0]
    
    opt.init_fom_eval()
    
    options_float = ['km', 'kr', 'pf','pop mult', 'pop size',\
                    'max generations', 'max generation mult',\
                    'sleep time', 'max log elements',\
                    'autosave interval',\
                    'parallel processes', 'parallel chunksize', 
                    'allowed fom discrepancy']
    set_float = [opt.km, opt.kr,opt.pf,
                opt.pop_mult,\
                opt.pop_size,\
                opt.max_generations,\
                opt.max_generation_mult,\
                opt.sleep_time,\
                opt.max_log, \
                opt.autosave_interval,\
                opt.processes,\
                opt.chunksize,\
                opt.fom_allowed_dis
                ]
    
    options_bool = ['use pop mult', 'use max generations',
                    'use start guess', 'use boundaries', 
                    'use parallel processing', 'use autosave',
                    ]
    set_bool = [ opt.use_pop_mult,
                opt.use_max_generations,
                opt.use_start_guess,
                opt.use_boundaries,
                opt.use_parallel_processing,
                opt.use_autosave,
                ]
    
    # Make sure that the config is set
    if config:
        # Start witht the float values
        for index in range(len(options_float)):
            try:
                val = config.set('solver', options_float[index],\
                                    set_float[index])
            except io.OptionError, e:
                print 'Could not locate save solver.' +\
                    options_float[index]
                
            # Then the bool flags
            for index in range(len(options_bool)):
                try:
                    val = config.set('solver',\
                                        options_bool[index], set_bool[index])
                except io.OptionError, e:
                    print 'Could not write option solver.' +\
                        options_bool[index]
                    
            try:
                config.set('solver', 'create trial',\
                            opt.get_create_trial())
            except io.OptionError, e:
                print 'Could not write option solver.create trial'
    else:
        print 'Could not write config to file'
    ### end of block: save config
    
    # build outfile name
    outfile = infile
    outfile = outfile.replace('.gx','')
    #outfile = '%s_%s_%s_kr%.2f_km%.2f_pf%.2f_run%d.gx' % (outfile, trial,fom, kr, km, pf,iter)
    outfile=outfile+fom+str(iter)+'_ran.gx'
    #outfile = '%s_%s_run%d.gx' % (outfile, TAG, iter)
    if rank==0:
        print 'Saving the initial model to %s'%outfile
        io.save_gx(outfile, mod, opt, config)
        
        print ''
        print 'Settings:'
        print '---------'
        
        print 'Number of fit parameters    = %s' % len(opt.best_vec)
        print 'FOM function                = %s' % mod.fom_func.func_name
        print ''
        print 'opt.km                      = %s' % opt.km
        print 'opt.kr                      = %s' % opt.kr
        print 'opt.create_trial            = %s' % opt.create_trial.im_func
        print ''
        print 'opt.use_parallel_processing = %s' % opt.use_parallel_processing
        print ''
        print 'opt.use_max_generations     = %s' % opt.use_max_generations
        print 'opt.max_generation_mult     = %s' % opt.max_generation_mult
        print 'opt.max_generations         = %s' % opt.max_generations
        print 'opt.max_gen                 = %s' % opt.max_gen
        print 'opt.max_log                 = %s' % opt.max_log
        print ''                          
        print 'opt.use_start_guess         = %s' % opt.use_start_guess
        print 'opt.use_boundaries          = %s' % opt.use_boundaries 
        print 'opt.use_autosave            = %s' % opt.use_autosave
        print 'opt.autosave_interval       = %s' % opt.autosave_interval
        print ''
        print 'opt.pop_size                = %s' % opt.pop_size       
        print 'opt.use_pop_mult            = %s' % opt.use_pop_mult   
        print 'opt.pop_mult                = %s' % opt.pop_mult       
        print 'opt.n_pop                   = %s' % opt.n_pop          
        print ''
        print '--------'
        print ''
        
        
        # To start the fitting
        print 'Fitting starting...'
    if rank==0:t1 = time.time()
    if rank==0:opt.text_output('Calculating start FOM ...')
    opt.running = True
    opt.error = False
    opt.n_fom = 0

    # Old leftovers before going parallel, rank 0 calculate fom vec and distribute to the other processors
    left,right=find_boundary(size,len(opt.pop_vec),rank)
    tmp_fom_vec=[opt.calc_fom(vec) for vec in opt.pop_vec[left:right+1]]
    comm.Barrier()
    tmp_fom_vec=comm.gather(tmp_fom_vec,root=0)
    if rank==0:
        tmp_fom_list=[]
        for i in list(tmp_fom_vec):
            tmp_fom_list=tmp_fom_list+i
        tmp_fom_vec=tmp_fom_list
    tmp_fom_vec=comm.bcast(tmp_fom_vec,root=0)
    opt.fom_vec=tmp_fom_vec
    """
    if rank==0:
        opt.fom_vec = [opt.calc_fom(vec) for vec in opt.pop_vec]
        tmp_fom_vec = opt.fom_vec
    tmp_fom_vec=comm.bcast(tmp_fom_vec,root=0)
    if rank!=0:
        opt.fom_vec=tmp_fom_vec
    """
    [opt.par_evals.append(vec, axis = 0)\
                for vec in opt.pop_vec]
    [opt.fom_evals.append(vec) for vec in opt.fom_vec]
    best_index = argmin(opt.fom_vec)
    opt.best_vec = copy(opt.pop_vec[best_index])
    opt.best_fom = opt.fom_vec[best_index]
    if len(opt.fom_log) == 0:
        opt.fom_log = r_[opt.fom_log,\
                            [[len(opt.fom_log),opt.best_fom]]]
    # Flag to keep track if there has been any improvemnts
    # in the fit - used for updates
    opt.new_best = True
    
    if rank==0:opt.text_output('Going into optimization ...')
    opt.plot_output(opt)
    opt.parameter_output(opt)
    
    comm.Barrier()
    
    t_mid=datetime.now()
    
    gen = opt.fom_log[-1,0] 
    
    if rank==0:
        mean_speed=0
        speed_inc=0.
    for gen in range(int(opt.fom_log[-1,0]) + 1, opt.max_gen\
                                + int(opt.fom_log[-1,0]) + 1):
        if opt.stop:
            break
        if rank==0:
            t_start = time.time()
            speed_inc=speed_inc+1.
        opt.init_new_generation(gen)
        
        # Create the vectors who will be compared to the 
        # population vectors
        #here rank 0 create trial vector and then broacast to the other processors 
        if rank==0:
            [opt.create_trial(index) for index in range(opt.n_pop)]
            tmp_trial_vec=opt.trial_vec
        else:
            tmp_trial_vec=0
        tmp_trial_vec=comm.bcast(tmp_trial_vec,root=0)
        opt.trial_vec=tmp_trial_vec
        #each processor only do a segment of trial vec
        opt.eval_fom()
        tmp_fom=opt.trial_fom
        comm.Barrier()
        #collect foms and reshape them and set the completed tmp_fom to trial_fom
        tmp_fom=comm.gather(tmp_fom,root=0)
        if rank==0:
            tmp_fom_list=[]
            for i in list(tmp_fom):
                tmp_fom_list=tmp_fom_list+i
            tmp_fom=tmp_fom_list
        tmp_fom=comm.bcast(tmp_fom,root=0)
        opt.trial_fom=np.array(tmp_fom).reshape(opt.n_pop,)
        
        # Calculate the fom of the trial vectors and update the population
        
        [opt.update_pop(index) for index in range(opt.n_pop)]
        
        # Add the evaluation to the logging
        [opt.par_evals.append(vec, axis = 0)\
                for vec in opt.trial_vec]
        [opt.fom_evals.append(vec) for vec in opt.trial_fom]
        
        # Add the best value to the fom log
        opt.fom_log = r_[opt.fom_log,\
                            [[len(opt.fom_log),opt.best_fom]]]
        
        if gen==1:
            # Let the model calculate the simulation of the best.
            sim_fom = opt.calc_sim(opt.best_vec)
    
            # Sanity of the model does the simualtions fom agree with
            # the best fom
            if rank==0:
                if abs(sim_fom - opt.best_fom) > opt.fom_allowed_dis:
		    print sim_fom,opt.best_fom
                    opt.text_output('Disagrement between two different fom'
                                    ' evaluations')
                    opt.error = ('The disagreement between two subsequent '
                                'evaluations is larger than %s. Check the '
                                'model for circular assignments.'
                                %opt.fom_allowed_dis)
                    break
        
        # Update the plot data for any gui or other output
        opt.plot_output(opt)
        opt.parameter_output(opt)
        
        # Let the optimization sleep for a while
        time.sleep(opt.sleep_time)
        
        # Time measurent to track the speed
        if rank==0:
            t = time.time() - t_start
            if t > 0:
                speed = opt.n_pop/t
                mean_speed=mean_speed+speed
            else:
                speed = 999999
            opt.text_output('FOM: %.3f Generation: %d Speed: %.1f'%\
                                (opt.best_fom, gen, speed))
        
        opt.new_best = False
        # Do an autosave if activated and the interval is coorect
        if rank==0 and gen%opt.autosave_interval == 0 and opt.use_autosave:

            opt.autosave()
	if gen%opt.autosave_interval==0:
	    
	    std_val=std(opt.fom_log[:,1][-200:])
	    if std_val<2:
		if rank==0:
		    opt.text_output('std='+str(std_val))
	        break
            else:
		if rank==0:
                    opt.text_output('std='+str(std_val))
            	    #calculate the error bar for parameters
            	    n_elements = len(opt.start_guess)
                    #print 'Number of elemets to calc errobars for ', n_elements
                    cum_N=0
                    for index in range(n_elements):
                        # calculate the error
                        # TODO: Check the error bar buisness again and how to treat
                        # Chi2
                        #print "senor",self.fom_error_bars_level
                        try:
                            (error_low, error_high) = opt.calc_error_bar(index, 1.05)
                        except:
                            break
                        error_str = '(%.3e, %.3e)'%(error_low, error_high)
                        while mod.parameters.get_value(index+cum_N,2)!=True:
                            cum_N=cum_N+1
                        mod.parameters.set_value(index+cum_N,5,error_str)
	            opt.autosave()

	    
    if rank==0:
        if not opt.error:
            opt.text_output('Stopped at Generation: %d after %d fom evaluations...'%(gen, opt.n_fom))
        
    # Lets clean up and delete our pool of workers

    opt.eval_fom = None
    
    # Now the optimization has stopped
    opt.running = False
    
    # Run application specific clean-up actions
    opt.fitting_ended(opt)
    
    t_end=datetime.now()
    
    if rank==0:
        t2 = time.time()
        print 'Fitting finsihed!'
        print 'Time to fit: ', (t2-t1)/60., ' min'
    
        print 'Updating the parameters'
    mod.parameters.set_value_pars(opt.best_vec)
    if rank==0:
    #calculate the error bar for parameters
       n_elements = len(opt.start_guess)
       #print 'Number of elemets to calc errobars for ', n_elements
       cum_N=0
       for index in range(n_elements):
           # calculate the error
           # TODO: Check the error bar buisness again and how to treat 
           # Chi2 
           #print "senor",self.fom_error_bars_level
           try:
              (error_low, error_high) = opt.calc_error_bar(index, 1.05)
           except:
               break
           error_str = '(%.3e, %.3e)'%(error_low, error_high)
	   while mod.parameters.get_value(index+cum_N,2)!=True:
	       cum_N=cum_N+1
	   mod.parameters.set_value(index+cum_N,5,error_str)
		


    if rank==0:
        print 'Saving the fit to %s'%outfile
        io.save_gx(outfile, mod, opt, config)
    
        print 'finished current fit'
        #if you want to create the logfile, uncomment following lines
        """
        fid = open(logfile,'a')
        fid.write('%-14s %4.2f %4.2f %4d %12.6g\n' % \
                    (fom, km, kr, iter, opt.fom_log[-1][1]))
        fid.close
        """
#t_mid-t_start_0 is the headover time before fitting starts, this headover time depend on # of cups used and can be up to 2hr in the case of using 100 cup chips
    if rank==0:
        print "run starts @",str(t_start_0)
        print "fitting starts @",str(t_mid)
        print "run stops @",str(t_end)
        print "headover time is ",str(t_mid-t_start_0)
        print "fitting time is ",str(t_end-t_mid)
        print 'Fitting sucessfully finished with mean speed of ',mean_speed/speed_inc
    else:
        pass
