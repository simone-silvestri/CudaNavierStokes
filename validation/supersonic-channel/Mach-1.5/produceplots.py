import matplotlib.pyplot as plt
import pandas as pd


cm = pd.read_csv("data/coleman.mean",sep=' ',header=None)
cm.columns = ['y','rm','pm','tm','wm','um','vm','mm','tmf','wmf','umf','vmf']
cm['y'] = cm['y']+1

cf = pd.read_csv("data/coleman.fluc",sep=' ',header=None)
cf.columns = ['y','wwf','uuf','vvf','uwf','ttf','vtf','wwr','uur','vvr','uwr','ttr','vtr']
cf['y'] = cf['y']+1

mf = pd.read_csv("data/present.fluc",sep=' ',header=None)
mf.columns = ['y','rrr','uuf','vvf','wwf','uur','vvr','wwr','eer','hhf','hhr','ttr','ppr','mmr']

mm = pd.read_csv("data/present.mean",sep=' ',header=None)
mm.columns = ['y','rm','umf','vmf','wmf','um','vm','wm','em','hmf','hm','tm','pm','mm']
mm['mm'] = mm['mm']*3000

fig, ax = plt.subplots()

cf.plot('y','wwr',ax=ax)
mf.plot('y','wwr',ax=ax)
cf.plot('y','uur',ax=ax)
mf.plot('y','uur',ax=ax)
cf.plot('y','vvr',ax=ax)
mf.plot('y','vvr',ax=ax)
plt.savefig('results/flucrey.eps', format='eps')

fig, ax = plt.subplots()

cm.plot('y','rm',ax=ax)
mm.plot('y','rm',ax=ax)
cm.plot('y','tm',ax=ax)
mm.plot('y','tm',ax=ax)
cm.plot('y','mm',ax=ax)
mm.plot('y','mm',ax=ax)
plt.savefig('results/prop.eps', format='eps')

fig, ax = plt.subplots()

cm.plot('y','wm',ax=ax)
mm.plot('y','wm',ax=ax)
plt.savefig('results/velrey.eps', format='eps')

fig, ax = plt.subplots()

cf.plot('y','ttr',ax=ax)
mf.plot('y','ttr',ax=ax)
plt.savefig('results/fluctemp.eps', format='eps')
