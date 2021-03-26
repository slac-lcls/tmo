import typing
import numpy as np
import pyqtgraph as pg
import ami.graph_nodes as gn
from amitypes import Array1d, Array2d
from ami.flowchart.library.common import CtrlNode
from ami.flowchart.library.DisplayWidgets import Histogram2DWidget
from pyqtgraph import QtCore
from psana.pop.POP import POP as psanaPOP
from psalg_ext import peak_finder_algos
from psana.hexanode.HitFinder import HitFinder
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d





class POPProc(object):

    def __init__(self, args):
        self.args = args
        self.accum_num = args.pop('accum_num', 1)
        self.normalize_dist = args.pop('normalizeDist', True)
        self.s = [1 if args.pop('Q'+str(i+1), True) else 0 for i in range(4)]
        self.proc = None
        self.counter = 0

    def __call__(self, img):

        if self.proc is None:
            self.img = np.zeros(img.shape)
            self.proc = psanaPOP(img=img, **self.args)

        self.img += img
        self.counter += 1

        if self.counter % self.accum_num == 0:
            self.proc.Peel(self.img,s = self.s)
            self.slice_img = self.proc.GetSlice()
            self.rbins, self.distr = self.proc.GetRadialDist()
            self.ebins, self.diste = self.proc.GetEnergyDist()
           # self.betas = self.GetBetas()
            self.counter = 0

            if self.normalize_dist:
                self.distr = self.distr/self.distr.max()
                self.diste = self.diste/self.diste.max()

        return self.slice_img, self.rbins[:-1], self.distr,self.ebins[:-1], self.diste


class POP(CtrlNode):

    """
    psana POP
    """

    nodeName = "POP"
      

    uiTemplate = [('RBFs_fnm', 'text'),
                  ('lmax', 'intSpin', {'value': 4, 'values': ['2', '4', '6', '8', '10', '12']}),
                  ('reg', 'doubleSpin', {'value': 0}),
                  ('alpha', 'doubleSpin', {'value': 0.0013}),
                  ('X0', 'intSpin', {'value': 274}),
                  ('Y0', 'intSpin', {'value': 276}),
                  ('Rmax', 'intSpin', {'value': 235}),
                  ('edge_w', 'intSpin', {'value': 10}),
                  ('accum_num', 'intSpin', {'value': 1, 'min': 0}),
                  ('normalizeDist', 'check', {'checked': True}),
                  ('Q1', 'check', {'checked': True}),
                  ('Q2', 'check', {'checked': True}),
                  ('Q3', 'check', {'checked': True}),
                  ('Q4', 'check', {'checked': True})]

    def __init__(self, name):
        super().__init__(name, terminals={'Image': {'io': 'in', 'ttype': Array2d},
                                          'sliceImg': {'io': 'out', 'ttype': Array2d},
                                          'Rbins': {'io': 'out', 'ttype': Array1d},
                                          'DistR': {'io': 'out', 'ttype': Array1d},                                        
                                          'Ebins': {'io': 'out', 'ttype': Array1d},
                                          'DistE': {'io': 'out', 'ttype': Array1d}})

    def to_operation(self, inputs, conditions={}):
        outputs = self.output_vars()

        node = gn.Map(name=self.name()+"_operation",
                      condition_needs=conditions,
                      inputs=inputs, outputs=outputs, parent=self.name(),
                      func=POPProc(self.values))
        return node
        

class QUADProc(object):

    def __init__(self, args):
    
        self.args = args
        self.HF = HitFinder(self.args)
        self.channels = {'mcp':0,'x1':1,'x2':2,'y1':3,'y2':4}

    def __call__(self, times, wfs):  
        
        ts = times[self.args['ind1']:self.args['ind2']]
        heights = {}
       
        for k in self.channels.keys():
            if k =='mcp':
                heights[k] = np.max(wfs[self.channels[k]][-1000:])+self.args[k+'_thresh']
            else:
                heights[k] = np.min(-wfs[self.channels[k]][-1000:])+self.args[k+'_thresh']           

    
    
        t_pks = {}
        for i,k in enumerate(self.channels.keys()):

            if k=='mcp':
                ind_t_pks,_ = find_peaks(gaussian_filter1d(wfs[self.channels[k]][self.args['ind1']:self.args['ind2']],10),
                               height=heights[k],prominence=self.args[k+'_prom'])
            else:
                ind_t_pks,_ = find_peaks(gaussian_filter1d(-wfs[self.channels[k]][self.args['ind1']:self.args['ind2']],10),
                                     height=heights[k],prominence=self.args[k+'_prom'])                        
            t_pks[k] = ts[ind_t_pks[:min(len(ind_t_pks),self.args['max_pknum'])]]*1e9    

        self.HF.FindHits(t_pks['mcp'],t_pks['x1'],t_pks['x2'],t_pks['y1'],t_pks['y2'])
        xs1,ys1,ts1 = self.HF.GetXYT()        

        return xs1,ys1,ts1   


class QUAD(CtrlNode):

    """
    peak_finder+psana_HitFinder
    """

    nodeName = "QUAD"


    uiTemplate = [('ind1', 'intSpin', {'value': 0}),
                  ('ind2', 'intSpin', {'value': 60000}), 
                  ('max_pknum', 'intSpin', {'value': 30}), 
                  ('mcp_thresh', 'doubleSpin', {'value': 2}),
                  ('x1_thresh', 'doubleSpin', {'value': 2}),
                  ('x2_thresh', 'doubleSpin', {'value': 2}),  
                  ('y1_thresh', 'doubleSpin', {'value': 2}),
                  ('y2_thresh', 'doubleSpin', {'value': 2}), 
                  ('mcp_prom', 'doubleSpin', {'value': 5}),                   
                  ('x1_prom', 'doubleSpin', {'value': 20}),
                  ('x2_prom', 'doubleSpin', {'value': 20}),  
                  ('y1_prom', 'doubleSpin', {'value': 20}),
                  ('y2_prom', 'doubleSpin', {'value': 20}),                                                       
                  ('runtime_u', 'doubleSpin', {'value': 185}),
                  ('runtime_v', 'doubleSpin', {'value': 185}),
                  ('tsum_avg_u','doubleSpin', {'value': 189}),
                  ('tsum_avg_v', 'doubleSpin', {'value': 195.5}),    
                  ('tsum_hw_u', 'doubleSpin', {'value': 10}),
                  ('tsum_hw_v','doubleSpin', {'value': 10}),                                
                  ('f_u', 'doubleSpin', {'value': 0.65}),
                  ('f_v', 'doubleSpin', {'value': 0.65}),
                  ('Rmax', 'doubleSpin', {'value': 65})]
                  
                  # entry point must be called func
    

    def __init__(self, name):
        super().__init__(name, terminals={'Times': {'io': 'in', 'ttype': Array1d},
                                          'WFs': {'io': 'in', 'ttype': Array2d},
                                          'X': {'io': 'out', 'ttype': Array1d},
                                          'Y': {'io': 'out', 'ttype': Array1d},
                                          'T': {'io': 'out', 'ttype': Array1d}})

    def to_operation(self, inputs, conditions={}):
        outputs = self.output_vars()

        node = gn.Map(name=self.name()+"_operation",
                      condition_needs=conditions,
                      inputs=inputs, outputs=outputs, parent=self.name(),
                      func=QUADProc(self.values))
        return node                
