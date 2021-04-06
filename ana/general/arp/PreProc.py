import os,sys,h5py

import numpy as np

from psana import DataSource


class PreProc:
    
    def __init__(self,params):

        self.params = params 
        self.params['hdf5'] = self.params['hdf5']+str(self.params['run_num'])+'.h5'


    def Process(self):   

        if 'max_evt' in self.params.keys():
            ds = DataSource(exp=self.params['exp'],run=self.params['run_num'],max_events=self.params['max_evt']) 
        else:
            ds = DataSource(exp=self.params['exp'],run=self.params['run_num'])             

        myrun = next(ds.runs())
        hsd = myrun.Detector('hsd')   

        det_evt = myrun.Detector('timing')
        det_gmd = myrun.Detector('gmd')
        det_xgmd = myrun.Detector('xgmd')
        det_ebm = myrun.Detector('ebeam')


        if os.path.isfile(self.params['hdf5']):
            os.remove(self.params['hdf5'])



        smd = ds.smalldata(filename=self.params['hdf5'], batch_size=self.params['refresh_num'])

        evt_dict = {}
        sevt_dict = {}


        sinit=None

        for nevt,evt in enumerate(myrun.events()):
            
            try:

                # Machine data
                evt_dict['evt'] = np.array(det_evt.raw.eventcodes(evt),dtype=np.int)
                evt_dict['gmd'] = det_gmd.raw.energy(evt)*1000
                evt_dict['xgmd'] = det_xgmd.raw.energy(evt)*1000
                evt_dict['pho'] = det_ebm.raw.ebeamPhotonEnergy(evt)
                evt_dict['ebm_l3'] = det_ebm.raw.ebeamL3Energy(evt)

                ##Ion data
                wfs = hsd.raw.waveforms(evt)
                wf_mcp = wfs[0][0].astype(np.float)
                
                if sinit == None:
                    sevt_dict['tof'] = np.zeros_like(wf_mcp) 
                    sinit = 1

                

                sevt_dict['tof'] += wf_mcp
                
                smd.event(evt, evt_dict)                


            except Exception as e:
                print(e)

            print('Event Num:',nevt) 
   

        if smd.summary:
            smd.sum(sevt_dict['tof'])
            smd.save_summary({'sig_sum':sevt_dict})
        smd.done()

if __name__ == "__main__":
    os.environ['PS_SRV_NODES']='1'
    os.environ['EXPERIMENT']='tmolw5618'
    os.environ['RUN_NUM']='16'
    exp = os.getenv('EXPERIMENT')
    run_num = os.getenv('RUN_NUM')
    params = {'exp':exp,'run_num':int(run_num),'refresh_num':100,'hdf5':'/reg/d/psdm/tmo/tmolw5618/results/xiangli/test_run','max_evt':100}
    print(params)
    prep = PreProc(params)
    prep.Process() 
