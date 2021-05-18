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
params = {'exp':exp,'run_num':int(run_num),'refresh_num':100,'hdf5':'/reg/d/psdm/tmo/tmolx0819/scratch/xiangli/v_run_tst_'}
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

    evt_dict = {}
    sevt_dict = {}


    sinit=None

    for nevt,evt in enumerate(myrun.events()):


        evt_dict['evt'] = np.array(det_evt.raw.eventcodes(evt),dtype=np.int)
            
        ##Ion data
        wfs = hsd.raw.waveforms(evt)
        if wfs is None:
            continue
        wf_mcp = wfs[0][0].astype(np.float)
        wf_mcp = wf_mcp - np.mean(wf_mcp[-100:])


        wf0 = np.zeros_like(wf_mcp)
        pks,prop = find_peaks(wf_mcp, prominence = 10)
        wf0[pks] = wf0[pks] + prop['prominences']

        if sinit==None:

            sinit = 1

            sevt_dict['wf'] = np.zeros_like(wf_mcp) 
            sevt_dict['wf_pks'] = np.zeros_like(wf_mcp) 
            sevt_dict['num_evt'] = 0


        sevt_dict['wf'] += wf_mcp
        sevt_dict['wf_pks'] += wf0
        sevt_dict['num_evt'] += 0


        smd.event(evt, evt_dict)                



        print('Event Num:',nevt) 

    if smd.summary:

        swf = smd.sum(sevt_dict['wf'])  
        swf_pks = smd.sum(sevt_dict['wf_pks'])  
        snum = smd.sum(sevt_dict['num_evt'])  

        smd.save_summary({'wf':swf,'wf_pks':swf_pks,'num_evt':snum})
 
    smd.done()    










