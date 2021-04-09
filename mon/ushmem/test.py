from psana import *

from psmon.plots import Image,XYPlot
from psmon import publish

import os
import numpy as np

from mpi4py import MPI
import time



global dat,t0,t500,num
dat = []
t0 = time.time()
num = 0

def CallBack(evt_dict):
    global dat,t0,t500,num

    img = Image(0,'atm',evt_dict['atm'])
    xy = XYPlot(0,'atm_proj',range(len(evt_dict['atm_proj'])),evt_dict['atm_proj'])
    #publish.local = True
    publish.send('ATM',img)
    publish.send('ATMP',xy)
    num += 1
    if num==500:
       t500 = time.time()
    elif num>500:
       print('t500:',t500-t0,'num:',num,'rank:',evt_dict['rank'],'tstamp:',evt_dict['tstamp'],'index:',evt_dict['index'])
    #dat.append(evt_dict['evr'])
    else:
       print('num:',num,'rank:',evt_dict['rank'],'tstamp:',evt_dict['tstamp'],'index:',evt_dict['index'])
   # if (evt_dict['index']+1)%5 == 0:
    #    print('index:',evt_dict['index'],'datl',len(dat),'dat0',dat[0])
     #   dat = []
if __name__=='__main__':
    t0 = time.time()
    os.environ['PS_SRV_NODES']='1'

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    ds = DataSource(shmem='tmo')
    smd = ds.smalldata(batch_size=5, callbacks=[CallBack])
    run = next(ds.runs())
    det_evr = run.Detector('timing')
    det_atm = run.Detector('tmo_atmopal')
    evt_dict = {}

    for nevt, evt in enumerate(run.events()):
        try:
            evt_dict['evr'] = np.array(det_evr.raw.eventcodes(evt),dtype=np.int)
            evt_dict['index'] = nevt
            evt_dict['rank'] = rank
            evt_dict['tstamp'] = evt.timestamp
            evt_dict['atm'] = det_atm.raw.image(evt)
            evt_dict['atm_proj'] = evt_dict['atm'].sum(0)
          #  for j in range(50):
           #     evt_dict['atmsum'] = det_atm.raw.image(evt).sum()
        except Exception as exc:
            print(exc)
        smd.event(evt,evt_dict)



    smd.done()
    

