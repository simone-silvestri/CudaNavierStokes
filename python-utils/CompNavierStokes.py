def CompNavierStokes(mx=128,my=128,mz=128, \
		     pRow=1,pCol=1, \
		     Lx=1.0,Ly=1.0,Lz=1.0, \
		     Re=1600,Pr=1.0,Ma=0.1,gam=1.4,visc=1,cfl=0.5, \
		     nUnifX=False,perX=True,forcing=False,stream=False,boundaryLayer=False, perturbed=False, \
                     stenA=4,stenV=4,restart=-1,nsteps=200,nfiles=1, \
		     checkCFL=10, checkBulk=10):
	
	import subprocess
	import os
	import sys
        import signal 
        import time
        from selfSimilarSol import selfSimilarSol

        if boundaryLayer:   
            selfSimilarSol(Ma=Ma,Pr=Pr,Re=Re,gam=gam)


	args= "make "
 	args+= "-f "
	args+= "makefiles/Makefile "
	args+= "mx_tot=-Dmx_tot="+str(mx) + " "
	args+= "my_tot=-Dmy_tot="+str(my) + " "
	args+= "mz_tot=-Dmz_tot="+str(mz) + " "
	args+= "Lx=-DLx="+str(Lx) + " "
	args+= "Ly=-DLy="+str(Ly) + " "
	args+= "Lz=-DLz="+str(Lz) + " "
	args+= "pRow=-DpRow="+str(pRow) + " "
	args+= "pCol=-DpCol="+str(pCol) + " "
	args+= "stencilSize=-DstencilSize="+str(stenA) + " "
	args+= "stencilVisc=-DstencilVisc="+str(stenV) + " "
 	args+= "Re=-DRe="+str(Re) + " "	
 	args+= "Pr=-DPr="+str(Pr) + " "	
 	args+= "Ma=-DMa="+str(Ma) + " "	
 	args+= "gam=-Dgam="+str(gam) + " "	
 	args+= "visc=-Dviscexp="+str(visc) + " "	
 	args+= "cfl=-DCFL="+str(cfl) + " "	
	args+= "checkCFLcondition=-DcheckCFLcondition="+str(checkCFL) + " "
	args+= "checkBulk=-DcheckBulk="+str(checkBulk) + " "

        if nUnifX:
		args+= "nonUniformX=-DnonUniformX=true" + " "
	else:
		args+= "nonUniformX=-DnonUniformX=false" + " "
		
        if perX:
		args+= "periodicX=-DperiodicX=true" + " "
	else:	
		args+= "periodicX=-DperiodicX=false" + " "

	if forcing:
		args+= "forcing=-Dforcing=true" + " "
	else:
		args+= "forcing=-Dforcing=false" + " "

 	if stream:
		args+= "useStream=-DuseStream=true" + " "
	else:
		args+= "useStream=-DuseStream=false" + " "

 	if boundaryLayer:
		args+= "boundaryLayer=-DboundaryLayer=true" + " "
	else:
		args+= "boundaryLayer=-DboundaryLayer=false" + " "

 	if perturbed:
		args+= "perturbed=-Dperturbed=true" + " "
	else:
		args+= "perturbed=-Dperturbed=false" + " "

	args+="restart=-Drestart="+str(restart) + " "
	args+="nsteps=-Dnsteps="+str(nsteps) + " "
	args+="nfiles=-Dnfiles="+str(nfiles)

        subprocess.call(["make","-f","makefiles/Makefile","clean"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	devnull = open(os.devnull, 'wb')
	p1 = subprocess.Popen(args=args, shell=True, stdout=devnull, stderr=subprocess.PIPE)
	output, error = p1.communicate()
        if not "" == error:
        	subprocess.call(["make","-f","makefiles/Makefile","clean"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
 		sys.exit("Error in compiling " + error)

  
	p1.wait()
        cores = pRow*pCol
 	p2 = subprocess.Popen(["mpirun","-np",str(cores),"./ns"])
	try:
	    p2.wait()
	except KeyboardInterrupt:
	    try:
               subprocess.call(["make","-f","makefiles/Makefile","clean"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	       p2.terminate()
	    except OSError:
	       pass
            p2.wait()

        subprocess.call(["make","-f","makefiles/Makefile","clean"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

