import os,sys,h5py,pickle,configparser

import numpy as np

from scipy.ndimage import gaussian_filter1d

from scipy.signal import find_peaks

from utils import config_name,rebin1dx,deconv_f

from psalg_ext import peak_finder_algos

from psana import DataSource


class PreProc:
    
    #parameter initilaization 
    def __init__(self,configf):

        config = configparser.ConfigParser()
        _ = config.read(configf) 

        ##params initializatoin for data processsing
        self.params = {}            
        for k0 in ['DataSource','Waveforms']:
            self.params[k0] = {}
            for k1 in config[k0].keys():  
                if k1=='run' or k1=='max_evt':
                    self.params[k0][k1] = int(config[k0][k1])
                else:
                    self.params[k0][k1] = config[k0][k1]
                    
        self.params['opal'] = {}       
        for k in config['opal'].keys():
            if k=='detname':
                self.params['opal'][k] = config['opal'][k]
            else:   
                self.params['opal'][k] = int(config['opal'][k])

        self.algos = peak_finder_algos(pbits=0)
        self.mask=np.ones([1024,1024], dtype=np.uint16)                    

                    
        self.params['WFconfig'] = {}
        chs = self.params['Waveforms']['chs'].split(',')

        for i,ch in enumerate(chs):
            self.params['WFconfig'][ch] = {}
            self.params['WFconfig'][ch]['name'] = ch
            for k in config[ch].keys():
                if k == 'channel' or k == 'maxnum':
                    self.params['WFconfig'][ch][k] = int(config[ch][k])
                elif k == 'polarity':
                    self.params['WFconfig'][ch][k] = config[ch][k]
                else:
                    self.params['WFconfig'][ch][k] = float(config[ch][k])
           

        self.params['h5nm'] = config['HDF5']['name']+'run'+str(self.params['DataSource']['run'])+'.h5'    

        self.refresh_num = int(config['Plots']['refreshNum'])  


        with open('resp_nobin.pkl','rb') as pkl:
            self.y_resp = pickle.load(pkl)        
            
        self.tof_dict = {'N2O+':[7440,7860],'O2+':[6520,6790],'N2+&NO+':[5900,6520],'H2O+':[4960,5330],
                    'O+':[4570,4890],'N+':[4150,4570],'O++':[3289,3608],'N++':[3000,3289],
                    'N+++':[2470,2780],'N++++':[2200,2400]}            

    ##Peakfinding and hit reconstruction with data from the TMO QUAD detector
    def Process(self):   

        if self.params['DataSource']['max_evt']==-1:
            ds = DataSource(exp=self.params['DataSource']['exp'],run=self.params['DataSource']['run']) 

        else:
            ds = DataSource(exp=self.params['DataSource']['exp'],run=self.params['DataSource']['run'],
                        max_events=self.params['DataSource']['max_evt'])   


        myrun = next(ds.runs())
        
        opal = myrun.Detector(self.params['opal']['detname'])
        hsd = myrun.Detector(self.params['Waveforms']['detname'])   

        det_evt = myrun.Detector('timing')
        det_gmd = myrun.Detector('gmd')
        det_xgmd = myrun.Detector('xgmd')
        det_ebm = myrun.Detector('ebeam')


        if os.path.isfile(self.params['h5nm']):
            os.remove(self.params['h5nm'])



        smd = ds.smalldata(filename=self.params['h5nm'], batch_size=self.refresh_num)

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
                evt_dict['evt70'] = evt_dict['evt'][70]


                ##Electron data
                img = opal.raw.image(evt)
                pks = self.algos.peak_finder_v4r3_d2(img, self.mask, 
                                                thr_low=self.params['opal']['thr_low'], 
                                                thr_high=self.params['opal']['thr_high'], 
                                                rank=self.params['opal']['rank'], 
                                                r0=self.params['opal']['r0'], 
                                                dr=self.params['opal']['dr'])
                coords = np.array([[pk.col,pk.row] for pk in pks])
                ele_hits = np.ones([self.params['opal']['maxnum'],2])*1e6


                evt_dict['ele_nhts'] = min(len(coords),self.params['opal']['maxnum'])
                ele_hits[:evt_dict['ele_nhts']] = coords[:evt_dict['ele_nhts'],:]
                evt_dict['ele_hits'] = ele_hits

                ##Ion data
                wfs = hsd.raw.waveforms(evt)
                factor = 1
                if sinit == None:
                    ts = wfs[channels['mcp']]['times']
                    ts = rebin1dx(ts,factor)*1e9
                    inds_dict = {}
                    ts_dict = {}
                    for ik,mk in enumerate(self.tof_dict.keys()):
                        inds_dict[mk] = ((ts>self.tof_dict[mk][0])&(ts<self.tof_dict[mk][1]))
                        ts_dict[mk] = ts[inds_dict[mk]]

                    sevt_dict['tof'] = np.zeros_like(ts) 
                    sinit = 1

                wf_mcp = wfs[channels['mcp']][0].astype(np.float)
                wf_mcp = rebin1d(wf_mcp,factor)
                wf_mcp = wf_mcp - np.mean(wf_mcp[-100:])

                wf_mcp = gaussian_filter1d(wf_mcp,3)

                recov = deconv_f(ts[1]-ts[0],wf_mcp,y_resp,5)
                wf_mcp = np.abs(recov)
                wf_mcp = wf_mcp[:min(len(ts),len(wf_mcp))]

              
                for ik,mk in enumerate(self.tof_dict.keys()):
                    evt_dict[mk+'_max'] = np.max(wf_mcp[inds_dict[mk]])
                    evt_dict[mk+'_sum'] = np.sum(wf_mcp[inds_dict[mk]])

                sevt_dict['tof'] += wf_mcp

                evt_dict['index'] = nevt
                smd.event(evt, evt_dict)

            except Exception as e:
                print(e)

            print('Event Num:',nevt,', Max Event Num:',self.params['DataSource']['max_evt']) 

            if nevt == self.params['DataSource']['max_evt']-1:
                break    

        if smd.summary:
            smd.sum(sevt_dict['tof'])
            smd.save_summary({'ion_sum':sevt_dict})
        smd.done()

if __name__ == "__main__":
    os.environ['PS_SRV_NODES']='1'
    configf = config_name()
    prep = PreProc(configf)
    prep.Process() 

   