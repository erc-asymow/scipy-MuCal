import ROOT
import numpy as np
import scipy as scipy
from scipy import linalg

f = ROOT.TFile.Open("/data/bendavid/muoncaldata/trackTree_test.root")

tree = f.tree

itrack = 0
for track in tree:
  print(track.trackEta)
  print(track.trackPt)
  print(track.trackCharge)
  
  #coodinate system is 
  # Global
  # q/|p|, lambda = pi/2-theta, phi, dxy, dsz
  
  #Local:
  #q/|p|, dx/dz, dy/dz, x,y
  
  nhits = track.trackH.size()
  
  #C is the 5x5 covariance matrix in global coords
  C = np.array(track.trackC).astype(np.float64)
  C = np.reshape(C,(-1,5,5))
  C = np.flip(C,axis=0)
  
  #F is the 5x5 transport matrix including energy loss
  F = np.array(track.trackF).astype(np.float64)
  F = np.reshape(F,(-1,5,5))
  F = np.swapaxes(F,-2,-1)
  F = np.flip(F,axis=0)
  #F[:,0,0] = 1./F[:,0,0]
  F = np.linalg.inv(F)
  
  #H is the 5x5 jacobian from global to local coordinates
  H = np.array(track.trackH).astype(np.float64)
  H = np.reshape(H,(-1,5,5))
  H = np.swapaxes(H,-2,-1)
  H = np.flip(H,axis=0)  
  H = np.linalg.inv(H)
  
  #Q is the 3x3 process noise in local coordinates (momentum part only)
  Q = np.array(track.trackQ).astype(np.float64)
  Q = np.reshape(Q,(-1,3,3))
  Q = np.flip(Q,axis=0)
  Qinv = np.linalg.inv(Q)
  #Qinv = Q
  #Qinv = Qinv[:-1]
  #print(Qinv.shape)
  #assert(0)
  
  #print(H)
  #print(F)
  #print(Qinv)
  #assert(0)
  
  x = np.array(track.localx).astype(np.float64)
  x = np.flip(x,axis=0)

  y = np.array(track.localy).astype(np.float64)
  y = np.flip(y,axis=0)
  
  xstate = np.array(track.localx_state).astype(np.float64)
  xstate = np.flip(xstate,axis=0)

  ystate = np.array(track.localy_state).astype(np.float64)
  ystate = np.flip(ystate,axis=0)
  
  dx = x - xstate
  dy = y - ystate
  

  
  xerrs = np.array(track.localxErr).astype(np.float64)
  xerrs = np.flip(xerrs,axis=0)
  
  yerrs = np.array(track.localyErr).astype(np.float64)
  yerrs = np.flip(yerrs,axis=0)
  
  xyerrs = np.array(track.localxyErr).astype(np.float64)
  xyerrs = np.flip(xyerrs,axis=0)
  
  xerrsinv = 1./xerrs
  yerrsinv = 1./yerrs
  
  
  #print(dx)
  #print(dy)
  #print(xerrs)
  #print(yerrs)
  #assert(0)
  

  #build "big" matrices
  Hfull = np.zeros((5*nhits,5*nhits),dtype=np.float64)
  Vinvfull = np.zeros((5*nhits,5*nhits),dtype=np.float64)
  Ffull = np.zeros((5*nhits,5*nhits),dtype=np.float64)
  Qinvfull = np.zeros((5*nhits,5*nhits),dtype=np.float64)
  dyfull = np.zeros((5*nhits,1),dtype=np.float64)
  Ifull = np.eye(5*nhits,dtype=np.float64)
  
  
  P = np.zeros((2*nhits,5*nhits),dtype=np.float64)
  
  eps = 1e-7
  
  for i in range(nhits):
    Hfull[5*i:5*(i+1),5*i:5*(i+1)] = H[i]
    P[2*i,5*i+3] = 1.
    P[2*i+1,5*i+4] = 1.
    

    
    #if xerrs[i]<1.:
      #Vinvfull[5*i+3,5*i+3] = xerrsinv[i]
      #dyfull[5*i+3,0] = dx[i]
    #if yerrs[i]<1.:
      #Vinvfull[5*i+4,5*i+4] = yerrsinv[i]
      #dyfull[5*i+4,0] = dy[i]
    #Vinvfull[5*i+3,5*i+3] = xerrsinv[i]
    #Vinvfull[5*i+4,5*i+4] = yerrsinv[i]
    Vtmp = np.array( [[xerrs[i], xyerrs[i]],
                    [xyerrs[i],yerrs[i]]], dtype=np.float64)
    Vinvfull[5*i+3:5*i+5,5*i+3:5*i+5] = np.linalg.inv(Vtmp)
    dyfull[5*i+3,0] = dx[i]
    dyfull[5*i+4,0] = dy[i]
    
    if i > 0:
      Ffull[5*i:5*(i+1),5*(i-1):5*i] = F[i-1]
      Qinvfull[5*i:5*i+3,5*i:5*i+3] = Qinv[i-1]
      #Qinvfull[5*i+3,5*i+3] = np.inf
      #Qinvfull[5*i+4,5*i+4] = np.inf
      #Qinvfull[5*i+3,5*i+3] = 1./eps**2
      #Qinvfull[5*i+4,5*i+4] = 1./eps**2
      
      #Qinvfull[5*(i-1)+3,5*(i-1)+3] = np.inf
      #Qinvfull[5*(i-1)+4,5*(i-1)+4] = np.inf
      #Qinvfull[5*(i-1)+3,5*(i-1)+3] = 1./1e-9**2
      #Qinvfull[5*(i-1)+4,5*(i-1)+4] = 1./1e-9**2
  
  
  #print(Ffull[5:10])
  #assert(0)
  
  #print(Hfull)
  #print(np.diag(Vinvfull))
  #print(Ffull)
  #print(Qinvfull)
  #assert(0)
  
  Cinvfull = Hfull.transpose() @ Vinvfull @ Hfull + (Ifull-Ffull).transpose() @ Hfull.transpose() @ Qinvfull @ Hfull @ (Ifull-Ffull)
  
  
  eigvals = np.linalg.eigvalsh(Cinvfull)
  #print(eigvals)
  #assert(0)
  
  d2chi2dx2 = 2.*Cinvfull
  d2chi2dxdlambda = P @ Hfull @ (Ifull-Ffull)
  
  Hbig = np.zeros((7*nhits,7*nhits),dtype=np.float64)
  
  Hbig[:5*nhits,:5*nhits] = d2chi2dx2  
  Hbig[5*nhits:,:5*nhits] = d2chi2dxdlambda
  Hbig[:5*nhits,5*nhits:] = d2chi2dxdlambda.transpose()
  
  #print(np.linalg.eigvalsh(Hbig))
  #assert(0)
  
  dchi2dx = -2.*Hfull.transpose() @ Vinvfull @ dyfull
  dchi2dlambda = np.zeros((2*nhits,1),dtype=np.float64)
  
  #print(dchi2dx)
  #assert(0)
  
  print(dchi2dx.shape)
  print(dchi2dlambda.shape)
  
  gbig = np.concatenate((dchi2dx,dchi2dlambda),axis=0)
  
  
  
  print(gbig.shape)
  
  #xlambda,res,rank,s = np.linalg.lstsq(Hbig,-gbig)
  xlambda = np.linalg.solve(Hbig,-gbig)
  #xlambda = linalg.solve(Hbig,gbig)
  #xlambda = linalg.lstsq(Hbig,gbig)[0]
  
  #diff = Hbig@xlambda-gbig
  #print("diff")
  #print(diff)
  
  #assert(0)
  
  dxout = xlambda[:5*nhits]
  lambdaout = xlambda[5*nhits:]
  
  dchi2dxout = -2.*Hfull.transpose()@Vinvfull@(dyfull-Hfull@dxout) + 2.*(Ifull-Ffull).transpose()@Hfull.transpose()@Qinvfull@Hfull@(Ifull-Ffull)@dxout + (Ifull-Ffull).transpose()@Hfull.transpose()@P.transpose()@lambdaout
  
  dchi2dlambdaout = P@Hfull@(Ifull-Ffull)@dxout
  
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
  
  dxinner = xlambda[:5,0]
  #dxinner = dxout[5:10,0]
  errsinnerorig = np.sqrt(np.diag(C[0]))
  
  
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
  Cfull = Cbig[:5*nhits,:5*nhits]
  
  
  #for i in range(nhits):
  for i in range(1):
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
  print(track.localxyErr)
  
    #print(C)
    #dyi = np.frombuffer(track.residual[i].data(),dtype=float)
    #print(dyi)
    
  #for i in range(nhits):
    #print(track.trackQ[i].size())
    #print(track.trackF[i].size())
    #print(track.trackH[i].size())
    #print(track.trackC[i].size())
    
  #print(dy)
  
  itrack += 1
  #break
  #if itrack>1:
    #break
