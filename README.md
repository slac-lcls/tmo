# tmo
Data analysis tools for the TMO instrument.
## Quick start
The tmo repo hosts software tools for online and offline data analysis of [TMO](https://lcls.slac.stanford.edu/instruments/neh-1-1) experiments. These tools arelargely facilited by the [LCLS-II software](https://github.com/slac-lcls/lcls2), especially its psana libray,  which is used to acquire and access the LCLS data, as well as to assist data analysis with its algorithms. One will be able to understand the tmo scripts and write their own data analysis script after going through the [psana confluence page](https://confluence.slac.stanford.edu/display/LCLSIIData/psana#psana-RunningFromSharedMemory). 

The repo is structured into three spaces - [ana](https://github.com/slac-lcls/tmo/tree/main/ana) for both general and instrument-specific scripts, [mon](https://github.com/slac-lcls/tmo/tree/main/mon) for online data analysis and [utils](https://github.com/slac-lcls/tmo/tree/main/utils) for handy library functions. 