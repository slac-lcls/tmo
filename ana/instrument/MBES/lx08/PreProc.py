import os,sys,h5py

import numpy as np

from psana import DataSource

from scipy.signal import find_peaks

from mpi4py import MPI




os.environ['PS_SRV_NODES']='1'
#    exp = os.getenv('EXPERIMENT')
#    run_num = os.getenv('RUN_NUM')

exp = 'tmolx0819'
run_num = '11'
h5_path = '/reg/d/psdm/tmo/tmolx0819/scratch/xiangli/'
params = {'exp':exp,'run_num':int(run_num),'refresh_num':100,'hdf5':h5_path+'run_'}
#print(params)
params = params 
params['hdf5'] = params['hdf5']+str(params['run_num'])+'.h5'


if os.path.isfile(params['hdf5']):
    os.remove(params['hdf5'])


if 'max_evt' in params.keys():
    ds = DataSource(exp=params['exp'],run=params['run_num'],max_events=params['max_evt']) 
else:
    ds = DataSource(exp=params['exp'],run=params['run_num'])             

smd = ds.smalldata(filename=params['hdf5'], batch_size=params['refresh_num'])

for myrun in ds.runs():
    hsd = myrun.Detector('hsd')   

    det_evt = myrun.Detector('timing')
    det_gmd = myrun.Detector('gmd')
    det_xgmd = myrun.Detector('xgmd')
    det_ebm = myrun.Detector('ebeam')

    det_las_t = myrun.Detector('las_fs14_target_time')
    det_las_d = myrun.Detector('las_fs14_target_time_dial')

    evt_dict = {}
    sevt_dict = {}
    maxNumPks = 100


    sinit=None
    ts = None
    ts_p = None

    for nevt,evt in enumerate(myrun.events()):
       # print('a:',nevt)
        
        tpks = np.array([np.nan]*maxNumPks)
        tpks_p = np.array([np.nan]*maxNumPks)        

        hpks = np.array([np.nan]*maxNumPks)
        hpks_p = np.array([np.nan]*maxNumPks)

        # Machine data
        evt_dict['evt'] = np.array(det_evt.raw.eventcodes(evt),dtype=int) #Event code
        #evt_dict['gmd'] = det_gmd.raw.energy(evt)#
        #evt_dict['xgmd'] = det_xgmd.raw.energy(evt)
        #evt_dict['pho'] = det_ebm.raw.ebeamPhotonEnergy(evt)
        #evt_dict['ebm_l3'] = det_ebm.raw.ebeamL3Energy(evt)
        #evt_dict['las_t'] = det_las_t(evt)
        #evt_dict['las_d'] = det_las_d(evt)
        #if evt_dict['las_d'] == None or evt_dict['gmd'] == None or evt_dict['xgmd'] == None:
        #    continue
        #evt_dict['gmd'] = evt_dict['gmd']*1000
        #evt_dict['xgmd'] = evt_dict['xgmd']*1000            

        #if evt_dict['las_t'] == None:
        #    evt_dict['las_t'] = np.nan   
        #print('b:',nevt)    
        ##Ion data
        wfs = hsd.raw.waveforms(evt)
        if wfs is None:
            continue
        wf_mcp = wfs[0][0].astype(float)
        
        if ts is None:
            ts = np.arange(0,len(wf_mcp))*0.167
        wf_mcp = wf_mcp - np.mean(wf_mcp[-100:])

        pks,prop = find_peaks(wf_mcp, prominence = 20)
        
        npks = min(len(pks),maxNumPks)
        
        
        tpks[:npks] = ts[pks][:npks]
        hpks[:npks] = prop['prominences'][:npks]

        evt_dict['n_pks'] = npks
        evt_dict['t_pks'] = tpks
        evt_dict['h_pks'] = hpks
        

        
###############################
        wfs = hsd.raw.padded(evt)
        if wfs is None:
            continue
        wf_mcp = wfs[0][0].astype(np.float)
        
        if ts_p is None:
            ts_p = np.arange(0,len(wf_mcp))*0.167
            
        wf_mcp = wf_mcp - np.mean(wf_mcp[-100:])
        
        
        if sinit==None:

            sinit = 1

            sevt_dict['wf'] = np.zeros_like(wf_mcp) 
            sevt_dict['wf_pks'] = np.zeros_like(wf_mcp) 
            sevt_dict['num_evt'] = 0


        wf0 = np.zeros_like(wf_mcp)
        pks,prop = find_peaks(wf_mcp, prominence = 20)
        wf0[pks] = wf0[pks] + prop['prominences']
        
        npks_p = min(len(pks),maxNumPks)
        tpks_p[:npks_p] = ts_p[pks][:npks_p]
        hpks_p[:npks_p] = prop['prominences'][:npks_p]
        
        evt_dict['n_pks_padded'] = npks_p
        evt_dict['t_pks_padded'] = tpks_p
        evt_dict['h_pks_padded'] = hpks_p        


        sevt_dict['wf'] += wf_mcp
        sevt_dict['wf_pks'] += wf0
        sevt_dict['num_evt'] += 1        

###############################
        smd.event(evt, evt_dict)                



        print('Event Num:',nevt) 

    if smd.summary:

        swf = smd.sum(sevt_dict['wf'])  
        swf_pks = smd.sum(sevt_dict['wf_pks'])  
        snum = smd.sum(sevt_dict['num_evt'])  

        smd.save_summary({'wf':swf,'wf_pks':swf_pks,'num_evt':snum})
 
    smd.done()    










