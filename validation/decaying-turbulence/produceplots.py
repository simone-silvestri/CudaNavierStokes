import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("data/reference-kinetic-energy", sep=',', header=None)
df.columns=['time','kinetic-energy']


df2 = pd.read_csv("data/present-solution", sep=' ', header=None)
df2.columns=['time','kinetic-energy','turbulent dissipation','time-step']


time = []
kin  = []
diss = []
for i in range(0,df2['time'].size):
	if df2['kinetic-energy'][i]!=0:
		time.append(df2['time'][i])
		kin.append(df2['kinetic-energy'][i]*0.5)
		diss.append(df2['turbulent dissipation'][i])


fig, ax = plt.subplots()

df.plot('time', 'kinetic-energy',ax=ax)
ax.plot(time,kin)
plt.savefig('kinetic-energy.eps', format='eps')

df = pd.read_csv("data/reference-turbulent-dissipation", sep=',', header=None)
df.columns=['time','turbulent dissipation']


fig2, ax2 = plt.subplots()

df.plot('time', 'turbulent dissipation',ax=ax2)
ax2.plot(time,diss)
plt.savefig('turbulent-dissipation.eps', format='eps')
