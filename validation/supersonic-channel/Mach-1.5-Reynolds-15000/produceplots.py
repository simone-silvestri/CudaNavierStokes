import matplotlib.pyplot as plt
import pandas as pd

rey = pd.read_csv("data/present.reynolds",sep=' ',header=None)
rey.columns = ['Ret','ut']
Ret = rey['Ret'][0]
ut  = rey['ut'][0]

cm = pd.read_csv("data/streams.Umean",sep=',',header=None)
cm.columns = ['y','um']

wf = pd.read_csv("data/streams.Ufluc",sep=',',header=None)
wf.columns = ['y','wwr']
uf = pd.read_csv("data/streams.Vfluc",sep=',',header=None)
uf.columns = ['y','uur']
vf = pd.read_csv("data/streams.Wfluc",sep=',',header=None)
vf.columns = ['y','vvr']

mm = pd.read_csv("data/present.mean",sep=' ',header=None)
mm.columns = ['y','rm','umf','vmf','wmf','um','vm','wm','em','hmf','hm','tm','pm','mm']

rw = Ret/ut/7362.5
#Ret = 489
#ut  = Ret/7362.5/rw 

tw = rw*ut**2

nm = mm #[mm['y']<1]

nm['y'] = nm['y']*Ret
nm['wm'] = nm['wm']/ut


mf = pd.read_csv("data/present.fluc",sep=' ',header=None)
mf.columns = ['y','rrr','uuf','vvf','wwf','uur','vvr','wwr','eer','hhf','hhr','ttr','ppr','mmr']

nf = mf #[nf['y']<1]

nf['y'] = nf['y']*Ret
nf['uur']/=tw 
nf['vvr']/=tw
nf['wwr']/=tw


fig, ax = plt.subplots()

wf.plot('y','wwr',ax=ax)
nf.plot('y','wwr',ax=ax)
uf.plot('y','uur',ax=ax)
nf.plot('y','uur',ax=ax)
vf.plot('y','vvr',ax=ax)
nf.plot('y','vvr',ax=ax)
##plt.xscale('log')
plt.savefig('results/flucrey.eps', format='eps')

fig, ax = plt.subplots()

cm.plot('y','um',ax=ax)
nm.plot('y','wm',ax=ax)
##plt.xscale('log')
plt.savefig('results/mean.eps', format='eps')

