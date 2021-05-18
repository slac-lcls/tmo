import os,sys,h5py

import numpy as np

from psana import DataSource

from scipy.signal import find_peaks

from mpi4py import MPI

class PreProc:
    
    def __init__(self,params):

        self.params = params 
        self.params['hdf5'] = self.params['hdf5']+str(self.params['run_num'])+'.h5'
        self.tst1 = 0


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

        det_las_t = myrun.Detector('las_fs14_target_time')
        det_las_d = myrun.Detector('las_fs14_target_time_dial')

        las_t = []
        las_d = []


        if os.path.isfile(self.params['hdf5']):
            os.remove(self.params['hdf5'])



        smd = ds.smalldata(filename=self.params['hdf5'], batch_size=self.params['refresh_num'])

        evt_dict = {}
        self.sevt_dict = {}
        test_dict = {}

        self.tst2 = 0
        self.tst3 = 0 


        sinit=None

        for nevt,evt in enumerate(myrun.events()):
            
            try:

                # Machine data
                evt_dict['evt'] = np.array(det_evt.raw.eventcodes(evt),dtype=np.int)
                evt_dict['gmd'] = det_gmd.raw.energy(evt)*1000
                evt_dict['xgmd'] = det_xgmd.raw.energy(evt)*1000
                evt_dict['pho'] = det_ebm.raw.ebeamPhotonEnergy(evt)
                evt_dict['ebm_l3'] = det_ebm.raw.ebeamL3Energy(evt)
                evt_dict['las_t'] = det_las_t(evt)
                evt_dict['las_d'] = det_las_d(evt)
                if evt_dict['las_d'] == None:
                    continue

                if evt_dict['las_t'] == None:
                    evt_dict['las_t'] = -1                



                ##Ion data
                wfs = hsd.raw.waveforms(evt)
                wf_mcp = wfs[0][0].astype(np.float)
                wf_mcp = wf_mcp - np.mean(wf_mcp[-100:])
                wf_mcp = -wf_mcp


                wf0 = np.zeros_like(wf_mcp)
                pks,prop = find_peaks(wf_mcp, prominence = 15)
                wf0[pks] = wf0[pks] + prop['prominences']
                
                if sinit==None:
                    test_dict['67'] = 0
                    sinit = 1
                if evt_dict['las_d'] not in las_d:
                    self.sevt_dict[str(evt_dict['las_d'])] = {}
                    las_d.append(evt_dict['las_d'])
                    for nm in ['tof_67','tof_68','tof_67n','tof_68n','ptof_67','ptof_68','ptof_67n','ptof_68n']:
                        self.sevt_dict[str(evt_dict['las_d'])][nm] = np.zeros_like(wf_mcp) 
                    self.sevt_dict[str(evt_dict['las_d'])]['67'] = 0
                    self.sevt_dict[str(evt_dict['las_d'])]['68'] = 0
                    
                    

                test_dict['67'] += 1.0
                self.tst1 += 1
                if evt_dict['evt'][161]==0 and evt_dict['xgmd']>5:
                    self.tst2+=1
                    if evt_dict['evt'][67]==1:
                        self.tst3 += 1
                        self.sevt_dict[str(evt_dict['las_d'])]['67'] += 1.0
                        self.sevt_dict[str(evt_dict['las_d'])]['tof_67'] += wf_mcp
                        self.sevt_dict[str(evt_dict['las_d'])]['tof_67n'] += wf_mcp/evt_dict['xgmd']
                        self.sevt_dict[str(evt_dict['las_d'])]['ptof_67'] += wf0
                        self.sevt_dict[str(evt_dict['las_d'])]['ptof_67n'] += wf0/evt_dict['xgmd']
                    if evt_dict['evt'][68]==1:
                        self.sevt_dict[str(evt_dict['las_d'])]['68'] += 1.0
                        self.sevt_dict[str(evt_dict['las_d'])]['tof_68'] += wf_mcp
                        self.sevt_dict[str(evt_dict['las_d'])]['tof_68n'] += wf_mcp/evt_dict['xgmd']
                        self.sevt_dict[str(evt_dict['las_d'])]['ptof_68'] += wf0
                        self.sevt_dict[str(evt_dict['las_d'])]['ptof_68n'] += wf0/evt_dict['xgmd']
                
                smd.event(evt, evt_dict)                


            except Exception as e:
                print(e)

          #  print('Event Num:',nevt) 
   
        print('pre_sum,rank:',self.params['rank'])
        if smd.summary:
            for ld in las_d:
                for nm in ['67','68','tof_67','tof_68','tof_67n','tof_68n','ptof_67','ptof_68','ptof_67n','ptof_68n']:
                    smd.sum(self.sevt_dict[str(ld)][nm])
            smd.sum(test_dict['67'])  
            smd.sum(self.tst1) 
            smd.sum(self.tst2) 
            smd.sum(self.tst3) 
            print('sum1,rank:',self.params['rank'])
            smd.save_summary(self.sevt_dict,test_dict,{'self.tst1':self.tst1,'self.tst2':self.tst2,'self.tst3':self.tst3})
            print('sum2,rank:',self.params['rank'])
        smd.done()
       

if __name__ == "__main__":
    os.environ['PS_SRV_NODES']='1'
#    exp = os.getenv('EXPERIMENT')
#    run_num = os.getenv('RUN_NUM')
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    exp = 'tmolw5618'
    run_num = '99'
    params = {'exp':exp,'run_num':int(run_num),'refresh_num':100,'hdf5':'/reg/d/psdm/tmo/tmolw5618/results/xiangli/v_run_','rank':rank}
    #print(params)
    prep = PreProc(params)
    prep.Process() 
    print('end,rank:',rank)









