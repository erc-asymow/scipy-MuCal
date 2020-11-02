import ROOT
import numpy as np
import scipy as scipy
from scipy import linalg

#f = ROOT.TFile.Open("/data/bendavid/muoncaldata/trackTree_test.root")
filename = "/data/bendavid/cmsswdevslc6/CMSSW_8_0_30/work/trackTree.root"
f = ROOT.TFile.Open(filename)

tree = f.tree

itrack = 0
for track in tree:
  
  itrack += 1
  ##if itrack<5:
    #continue;
  #if track.ninvalidHits>0:
    #continue
  #if abs(track.trackPt)>10.:
    #continue
  #if abs(track.trackPt)<100.:
    #continue

  #if (itrack<17):
    #continue
  
  print(track.trackEta)
  print(track.trackPt)
  print(track.trackPhi)
  print(track.trackCharge)
  
  #coodinate system is 
  # Global
  # q/|p|, lambda = pi/2-theta, phi, dxy, dsz
  
  #print(type(track.trackC))
  #print(type(track.trackH))
  #print(type(track.trackQ))
  #print(type(track.trackF))
  #assert(0)
  
  #Local:
  #q/|p|, dx/dz, dy/dz, x,y
  
  nhits = track.trackH.size()
  
  detector = np.array(track.detector)
  layer = np.array(track.layer)
  print(detector)
  print(layer)
  #assert(0)
  
  #C is the 5x5 covariance matrix in global coords
  C = np.array(track.trackC).astype(np.float64)
  C = np.reshape(C,(-1,5,5))
  #C = np.flip(C,axis=0)
  
  #F is the 5x5 transport matrix including energy loss
  F = np.array(track.trackF).astype(np.float64)
  F = np.reshape(F,(-1,5,5))
  
  #Ff = np.array(track.trackFf).astype(np.float64)
  #Ff = np.reshape(Ff,(-1,5,5))
  #print(Fr.shape)
  #print(Ff.shape)
  #assert(0)
  
  #F = np.swapaxes(F,-2,-1)
  #F = np.flip(F,axis=0)
  #F[:,0,0] = 1./F[:,0,0]
  #F = np.linalg.inv(F)
  #print(F)
  
  #assert(0)
  
  #H is the 5x5 jacobian from global to local coordinates
  H = np.array(track.trackH).astype(np.float64)
  H = np.reshape(H,(-1,5,5))
  
  Hprop = np.array(track.trackHprop).astype(np.float64)
  Hprop = np.reshape(Hprop,(-1,5,5))
  
  #Hf = np.array(track.trackHf).astype(np.float64)
  #Hf = np.reshape(Hf,(-1,5,5))
  
  #H = np.swapaxes(H,-2,-1)
  #H = np.linalg.inv(H)
  
  #print(H)
  #assert(0)
  
  #Q is the 3x3 process noise in local coordinates (momentum part only)
  Q = np.array(track.trackQ).astype(np.float64)
  Q = np.reshape(Q,(-1,3,3))
  
  #Qf = np.array(track.trackQf).astype(np.float64)
  #Qf = np.reshape(Qf,(-1,3,3))
  
  
  print(H[0])
  print(H[1])
  print(F[1])
  print(np.linalg.inv(Q[1]))
  assert(0)
  
  dimension = np.array(track.hitDimension)
  print(dimension)
  stereo = np.array(track.stereo)
  glued = np.array(track.glued)
  detid = np.array(track.detid)
  print(stereo)
  print(glued)
  print(detid)
  

  
  xerrs = np.array(track.localxErr).astype(np.float64)
  #xerrs = np.flip(xerrs,axis=0)
  
  yerrs = np.array(track.localyErr).astype(np.float64)
  #yerrs = np.flip(yerrs,axis=0)
  
  xyerrs = np.array(track.localxyErr).astype(np.float64)
  #xyerrs = np.flip(xyerrs,axis=0)
  
  print(xerrs)
  print(yerrs)
  print(xyerrs)
  

  
  xycors = xyerrs/np.sqrt(xerrs*yerrs)
  print(xycors)
  
  #assert(0)
  
  #assert(0)
  #assert90
  
  #Q = Q[1:]
  #Qinv = np.linalg.inv(Q)
  #Qinv = Q
  #Qinv = Q
  #Qinv = Qinv[:-1]
  #print(Qinv.shape)
  #print(Q)
  #assert(0)
  #assert(0)
  
  #print(H)
  ##print(F)
  #print(Qinv)
  #assert(0)
  
  
  ustate = np.array(track.updState).astype(np.float64)
  fstate = np.array(track.forwardPropState).astype(np.float64)
  bstate = np.array(track.backPropState).astype(np.float64)
  
  ustatelocal = np.array(track.updStateLocal).astype(np.float64)
  fstatelocal = np.array(track.fwdPredStateLocal).astype(np.float64)
  bstatelocal = np.array(track.bkgPredStateLocal).astype(np.float64)
  
  bpropstate = np.array(track.bkgupdPropState).astype(np.float64)
  
  
  #bstate = np.array(track.backPropState).astype(np.float64)
  #fstate = np.array(track.forwardPropState).astype(np.float64)
  dx0 = ustatelocal - bpropstate
  #print(dx0)
  #assert(0)
  dx0full = np.reshape(dx0,(5*nhits,1))
  

  #dx0 *= 0.
  #print(dx0)
  #assert(0)
  
  print(ustate)
  #print(pstate)
  print(dx0/ustate)
  #assert(0)
  
  #dx0[0] *= 0.
  #dx0[:,3:] *=0.
  #dx0 *= 0.
  
  #print(dx0)
  #assert(0)
  
  #xufull = np.reshape(ustate,(5*nhits,1))
  
  #Hinv = np.linalg.inv(H)
  
  
  
  #print("transport closure compare")
  
  #print(ustate)
  #assert(0)
  
  ##for i in range(nhits):
    ##print(Hinv[i] @ ustate[i])
  ##assert(0)
  
  #gu0 = Hinv[0] @ ustate[0]
  #gu0prop = F[0] @ Hinv[0] @ ustate[0]
  #gu1 = Hinv[1] @ ustate[1]
  ##bu1 = Hinv[1] @ bstate[1]
  #diff = gu1-gu0prop
  #print(gu0)
  #print(gu0prop)
  #print(gu1)
  ##print(bu1)
  #print(diff)
  #assert(0)
  
  
  #print(fstate)
  #print(bstate)
  #print(ustate)
  #assert(0)
  #print(dx0)
  
  #assert(0)
  #dx0[:,3:] *= 0.
  #dx0full *= 0.
  #print(dx0full)
  #assert(0)
  #print(dx0.shape)
  #assert(0)
  
  x = np.array(track.localx).astype(np.float64)
  #x = np.flip(x,axis=0)

  y = np.array(track.localy).astype(np.float64)
  #y = np.flip(y,axis=0)
  
  xstate = np.array(track.localx_state).astype(np.float64)
  #xstate = np.flip(xstate,axis=0)

  ystate = np.array(track.localy_state).astype(np.float64)
  #ystate = np.flip(ystate,axis=0)
  
  dx = x - xstate
  dy = y - ystate
  
  ymeas = np.array(track.curvpars)
  ymeas = ymeas[...,np.newaxis]
  print(ymeas.shape)
  print(ymeas[0].shape)
  print(ymeas)
  for iy in ymeas:
    if iy[2,0] < 0.:
      iy[1,0] *= -1.
      iy[2,0] += np.pi
      iy[3,0] *= -1.
      #iy[4,0] *= -1.
  #print(ymeas[:,1,:])
  print(ymeas)
  #assert(0)
  
  #phi = np.array(track.phi).astype(np.float64)
  #print(phi)
  #assert(0)
  ##print("transformed to local")
  #print(H@ymeas)
  #assert(0)
  
  #print(dx)
  #print(dy)
  
  #assert(0)
  

  
  stereo = np.array(track.stereo).astype(np.float64)
  glued = np.array(track.glued).astype(np.float64)
  
  #print(stereo)
  #print(glued)
  #assert(0)
  
  xerrsinv = 1./xerrs
  yerrsinv = 1./yerrs
  
  
  #print(dx)
  #print(dy)
  #print(xerrs)
  #print(yerrs)
  #assert(0)
  


  #build "big" matrices
  Hfull = np.zeros((5*nhits,5*nhits),dtype=np.float64)
  Hpropfull = np.zeros((5*nhits,5*nhits),dtype=np.float64)
  
  Vinvfull = np.zeros((5*nhits,5*nhits),dtype=np.float64)
  Ffull = np.zeros((5*nhits,5*nhits),dtype=np.float64)

  
  Frevfull = np.zeros((5*nhits,5*nhits),dtype=np.float64)
  Qinvfull = np.zeros((5*nhits,5*nhits),dtype=np.float64)
  Qinvfullalt = np.zeros((5*nhits,5*nhits),dtype=np.float64)
  dyfull = np.zeros((5*nhits,1),dtype=np.float64)
  yfull = np.zeros((5*nhits,1),dtype=np.float64)
  Ifull = np.eye(5*nhits,dtype=np.float64)
  #Ifull[:5] *= 2.
  #Ifull[:5] *= 2.
  x0full = np.zeros((5*nhits,1), dtype=np.float64)
  #dx0full = np.zeros((5*nhits,1), dtype=np.float64)
  
  P = np.zeros((2*nhits,5*nhits),dtype=np.float64)
  
  
  
  #eps = 1e-7
  #eps = 1e-14
  eps = 1e-5
  
  for i in range(nhits):
    Hfull[5*i:5*(i+1),5*i:5*(i+1)] = H[i]
    Hpropfull[5*i:5*(i+1),5*i:5*(i+1)] = Hprop[i]

    if dimension[i] == 2:
      Vtmp = np.array( [[xerrs[i], xyerrs[i]],
                    [xyerrs[i],yerrs[i]]], dtype=np.float64)
      Vinvfull[5*i+3:5*i+5,5*i+3:5*i+5] = np.linalg.inv(Vtmp)
    else:
      Vinvfull[5*i+3,5*i+3] = 1./xerrs[i]
    
    dyfull[5*i+3,0] = x[i] - ustatelocal[i,3]
    dyfull[5*i+4,0] = y[i] - ustatelocal[i,4]
    
    #x0full[5*i:5*i+5,0] = ustate[i]
    
    #if i>0:
      #Ffull[5*i:5*(i+1),5*(i+1):5*(i+2)] = F[i+1]
      #Qinvtmp = np.zeros((5,5),dtype=np.float64)
      #Qinvtmp[:3,:3] = np.linalg.inv(Q[i+1])
      #Qinvtmp[3,3] = 1./eps**2
      #Qinvtmp[4,4] = 1./eps**2
      #Qinvtmp = H[i+1].transpose() @ Qinvtmp @ H[i+1]
      #Qinvfull[5*i:5*i+5,5*i:5*i+5] = Qinvtmp
    
    #if i<(nhits-1):
      #Ffull[5*i:5*(i+1),5*(i+1):5*(i+2)] = F[i+1]
      #Qinvtmp = np.zeros((5,5),dtype=np.float64)
      #Qinvtmp[:3,:3] = np.linalg.inv(Q[i+1])
      #Qinvtmp[3,3] = 1./eps**2
      #Qinvtmp[4,4] = 1./eps**2
      #Qinvtmp = H[i+1].transpose() @ Qinvtmp @ H[i+1]
      #Qinvfull[5*i:5*i+5,5*i:5*i+5] = Qinvtmp
      
      #dx0full[5*i:5*i+5,0] = ustate[i] - fpropstate[i+1]
    
    if i > 0:
      Ffull[5*i:5*(i+1),5*(i-1):5*i] = F[i]
      Qinvtmp = np.zeros((5,5),dtype=np.float64)
      Qinvtmp[:3,:3] = np.linalg.inv(Q[i])
      Qinvtmp[3,3] = 1./eps**2
      Qinvtmp[4,4] = 1./eps**2
      Qinvfull[5*i:5*i+5,5*i:5*i+5] = Qinvtmp
      
      P[2*i,5*i+3] = 1.
      P[2*i+1,5*i+4] = 1.
      
      
      ##Qrinvfull[5*i:5*i+3,5*i:5*i+3] = np.linalg.inv(Qr[i])
      ##Qrinvfull[5*i+3,5*i+3] = 1./eps**2
      ##Qrinvfull[5*i+4,5*i+4] = 1./eps**2
      
      #Pr[2*i,5*i+3] = 1.
      #Pr[2*i+1,5*i+4] = 1.
    
    
    
    
      #Qinvfull[5*i+3,5*i+3] = np.inf
      #Qinvfull[5*i+4,5*i+4] = np.inf
      #Qinvfull[5*i,5*i] += 1./eps**2
      #Qinvfull[5*i+3,5*i+3] = 1./eps**2
      #Qinvfull[5*i+4,5*i+4] = 1./eps**2
      
      #Qinvfull[5*(i-1)+3,5*(i-1)+3] = np.inf
      #Qinvfull[5*(i-1)+4,5*(i-1)+4] = np.inf
      #Qinvfull[5*(i-1)+3,5*(i-1)+3] = 1./1e-9**2
      #Qinvfull[5*(i-1)+4,5*(i-1)+4] = 1./1e-9**2
  
    #if i<(nhits-1):
      #Frevfull[5*i:5*(i+1),5*(i+1):5*(i+2)] = np.linalg.inv(F[i+1])
      #Frevfull[5*(i+1):5*(i+2),5*i:5*(i+1)] = np.linalg.inv(F[i+1])
      
      #Qinvfullalt[5*i:5*i+3,5*i:5*i+3] = Qinv[i+1]
      
      #Qinvfullalt[5*i+3,5*i+3] = 1./eps**2
      #Qinvfullalt[5*i+4,5*i+4] = 1./eps**2
      
    #if i<(nhits-1):
      ##print(Fffull.shape)
      ##print(Fffull[5*i:5*(i+1),5*(i-1):5*i].shape)
      ##print(Ff[i].shape)
      #Fffull[5*i:5*(i+1),5*(i+1):5*(i+2)] = Ff[i]
      ##Fffull[5*i:5*(i+1),5*(i+1):5*(i+2)] = np.linalg.inv(Fr[i+1])
      #Qfinvfull[5*i:5*i+3,5*i:5*i+3] = np.linalg.inv(Qf[i])
      #Qfinvfull[5*i+3,5*i+3] = 1./eps**2
      #Qfinvfull[5*i+4,5*i+4] = 1./eps**2
      
  
  
  Hhit = 2.*Hfull.transpose() @ Vinvfull @ Hfull
  Hrms = 2.*(Ifull-Ffull).transpose() @ Hfull.transpose() @  Qinvfull @ Hfull @ (Ifull-Ffull)
  #Hrms = 2.*(Hfull - Hpropfull@Ffull).transpose() @ Qinvfull @ (Hfull - Hpropfull@Ffull)
  
  ghit = -2.*Hfull.transpose() @ Vinvfull @ dyfull
  grms = 2.*(Ifull-Ffull).transpose() @ Hfull.transpose() @ Qinvfull @ dx0full
  #grms = 2.*(Hfull - Hpropfull@Ffull).transpose() @ Qinvfull @ dx0full
  
  print(ghit)
  print(grms)
  #print(dx0full)
  #assert(0)
  
  #print(dx0full)
  ##print((Ifull-Ffull) @ x0full)
  #print(ustate[0])
  #print(ustate[1])
  #print(F[1])
  #print(F[1]@ustate[1])
  #print(fpropstate[1])
  #print(ustate[0] - F[1]@ustate[1])
  #assert(0)
  
  H = Hhit+Hrms
  g = ghit+grms
  
  Cfull = np.linalg.inv(0.5*H)
  dxout = np.linalg.solve(H,-g)
  dxout = np.reshape(dxout,(-1,5))

  dxout = np.reshape(dxout,(-1,5))
  #xout = bstate + dxout
  #xoutinner = xout[-1]
  
  #print(xoutinner)
  #print(bstate[-1])
  print(dxout[-1])
  #print(xout[-1])
  print(ustate[-1])
  
  errsinnerorig = np.sqrt(np.diag(C[-1]))
  
  dxinnerpull = dxout[-1]/errsinnerorig
  print("dxinnerpull")
  print(dxinnerpull)
  
  dxpull = np.reshape(dxout,(-1,))/np.sqrt(np.diag(Cfull))
  print("dxpull")
  print(dxpull)
  
  edmval = 0.5*g.transpose() @ Cfull @ g
  print("edmval", edmval)
  
  
  for i in range(nhits-1,nhits):
    Ci = Cfull[5*i:5*(i+1),5*i:5*(i+1)]
    Ciorig = C[i]
    #Ciorig = H[i].transpose()@C[i]@H[i]
    print(Ci)
    print(Ciorig)
  
  assert(0)
  
  P = P[2:]
  
  #print(dyfull)
  #assert(0)
  
  #print(Vinvfull)
  #assert(0)
  
  #Ffullalt = np.linalg.pinv(Ffull)
  #Ffull = 0.5*(Ffull+np.linalg.pinv(Ffull))
  #Ffull[0] *= 2.
  #Ffull[-1] *= 2.
  #Ffull = 0.5*(Ffull+Frevfull)
  
  #print(dx0full)
  #dx0full = np.linalg.inv(Hfull)@(Ifull-Ffull)@xufull
  #dx0full = (Ifull-Ffull)@np.linalg.inv(Hfull) @ xufull
  #dx0full = np.linalg.inv(Hfull) @ dx0full
  
  #dx0full *= 0.
  #print("dxfullclosure")
  #print(dx0full)
  #assert(0)
  
  #xlfull = xufull
  #xlfull = xufull
  
  
  #print("xufull")
  #print(xufull)
  
  #print("xstate")
  #print(xstate)
  #print("ystate")
  #print(ystate)  
  #assert(0)
  
  #print(Ffull[5:10])
  #assert(0)
  
  #print(Hfull)
  #print(np.diag(Vinvfull))
  #print(Ffull)
  #print(Qinvfull)
  #assert(0)
  
  #d2chi2dx2 = 2.*Hfull.transpose() @ Vinvfull @ Hfull + 2.*(Ifull-Ffull).transpose() @ Hfull.transpose() @ Qinvfull @ Hfull @ (Ifull-Ffull)
  
  d2chi2dx2hit = 2.*Hfull.transpose() @ Vinvfull @ Hfull
  d2chi2dx2mom = 2.*(Ifull-Ffull).transpose() @ Hfull.transpose() @ Qinvfull @ Hfull @ (Ifull-Ffull)
  #d2chi2dx2mom = 2.*(Hfull-Hpropfull@Ffull).transpose() @ Qinvfull @ (Hfull-Hpropfull@Ffull)
  d2chi2dx2 = d2chi2dx2hit + d2chi2dx2mom
  
  Cinvfull = 0.5*d2chi2dx2
  d2chi2dxdlambda = P @ Hfull @ (Ifull-Ffull)
  #d2chi2dxdlambda = P @ (Hfull-Hpropfull@Ffull)
  
  Hbig = np.zeros((7*nhits-2,7*nhits-2),dtype=np.float64)
  
  Hbig[:5*nhits,:5*nhits] = d2chi2dx2  
  Hbig[5*nhits:,:5*nhits] = d2chi2dxdlambda
  Hbig[:5*nhits,5*nhits:] = d2chi2dxdlambda.transpose()
  
  dchi2dxhit = -2.*Hfull.transpose() @ Vinvfull @ dyfull
  dchi2dxmom = 2.*(Ifull-Ffull).transpose()@Hfull.transpose()@Qinvfull @ dx0full
  #dchi2dxmom = 2.*(Hfull-Hpropfull@Ffull).transpose()@Qinvfull @ dx0full
  dchi2dx = dchi2dxhit + dchi2dxmom
  dchi2dlambda = P @ dx0full
  
  
  gbig = np.concatenate((dchi2dx,dchi2dlambda),axis=0)
  
  print(gbig.shape)
  
  #xlambda,res,rank,s = np.linalg.lstsq(Hbig,-gbig)
  xlambda = np.linalg.solve(Hbig,-gbig)
  #xlambda = linalg.solve(Hbig,gbig)
  #xlambda = linalg.lstsq(Hbig,gbig)[0]
  
  #edm = -0.5*gbig.transpose() @ xlambda
  #print("edmval", edm)
  
  #diff = Hbig@xlambda-gbig
  #print("diff")
  #print(diff)
  
  #assert(0)
  
  dxout = xlambda[:5*nhits]
  lambdaout = xlambda[5*nhits:]
  
  #dchi2dxout = -2.*Hrfull.transpose()@Vinvfull@(dyfull-Hrfull@dxout) + 2.*(Ifull-Ffull).transpose()@Hfull.transpose()@Qinvfull@Hfull@(Ifull-Ffull)@dxout + (Ifull-Ffull).transpose()@Hfull.transpose()@P.transpose()@lambdaout
  
  #dchi2dlambdaout = P@Hfull@(Ifull-Ffull)@dxout
  
  #print(dchi2dxout)
  #print(dchi2dlambdaout)
  #assert(0)
  
  #print(rank)
  #print(s)
  #print(s.shape)
  #print(xlambda.shape)
  ##assert(0)
  
  #print("xlambda")
  #print(xlambda)
  
  #dxinner = xlambda[:5,0]
  #dxinner = dxout[5:10,0]
  #errsinnerorig = np.sqrt(np.diag(C[0]))
  
  #dxout = np.linalg.solve(d2chi2dx2,-dchi2dx) + x0full
  
  print(dxout)
  
  dxinner = dxout[-5:,0]
  #dxinner = dxout[:5,0]
  errsinnerorig = np.sqrt(np.diag(C[-1]))
  #errsinnerorig = np.sqrt(np.diag(C[0]))
  
  print("xinner")
  print(dxinner)
  
  #print(Hinv[-1]@ustate[-1])
  #print(ustate[-1])
  print(ustate[0])
  theta = 2.*np.arctan(np.exp(-track.trackEta))
  print(track.trackCharge*np.sin(theta)/track.trackPt)
  
  print("dxinnerpull")
  dxinnerpull = dxinner/errsinnerorig
  
  print(dxinnerpull)
  

  
  
  #print(dchi2dx)
  
  #assert(0)
  
  #print(np.linalg.eigvalsh(Hbig))
  #assert(0)
  
  #Hbig *= 0.5
  ##print(Hbig[-1,-1])
  
  #print(Hbig)
  #assert(0)

  
  #Cinv = Hfull.transpose()@Vinvfull@Hfull 
  
  #print(Cinvfull)
  #assert(0)
  #print(Cinv.shape)
      
  #Cfull = np.linalg.inv(Cinvfull)
  Cbig = np.linalg.inv(0.5*Hbig)
  #Cbig = linalg.pinv(0.5*Hbig)
  #Cbig = np.linalg.pinv(0.5*Hbig)
  Cfull = Cbig[:5*nhits,:5*nhits]
  
  print("dxpull")
  dxpull = np.reshape(dxout,(-1,))/np.sqrt(np.diag(Cfull))
  print(dxpull)
  
  gsmall = gbig[:5*nhits]
  
  edmval = 0.5*gsmall.transpose() @ Cfull @ gsmall
  print("edmval", edmval)
  
  corqp01 = Cfull[0,-5]/np.sqrt(Cfull[0,0]*Cfull[-5,-5])
  print("corqp01", corqp01)
  diffqp01 = ustate[0,0] - ustate[-1,0]
  sigmaqp01 = np.sqrt(Cfull[0,0] + Cfull[-5,-5] - 2*Cfull[0,-5])
  print(diffqp01, sigmaqp01, diffqp01/sigmaqp01)
  
  #print(np.linalg.eigvalsh(Cfull))
  #assert(0)
  
  #Cfull = np.linalg.inv(Cinvfull)
  
  
  #for i in range(nhits):
  for i in range(nhits-1,nhits):
    Ci = Cfull[5*i:5*(i+1),5*i:5*(i+1)]
    Ciorig = C[i]
    #Ciorig = H[i].transpose()@C[i]@H[i]
    print(Ci)
    print(Ciorig)
    
  
  
  #print(C)
  #print(C.shape)
  #print(F.shape)
  #print(H.shape)
  #print(Q.shape)
  
  #print(track.localxErr)
  #print(track.localyErr)
  #print(track.localxyErr)
  
    #print(C)
    #dyi = np.frombuffer(track.residual[i].data(),dtype=float)
    #print(dyi)
    
  #for i in range(nhits):
    #print(track.trackQ[i].size())
    #print(track.trackF[i].size())
    #print(track.trackH[i].size())
    #print(track.trackC[i].size())
    
  #print(dy)
  
  break
  #if itrack>1:
    #break
