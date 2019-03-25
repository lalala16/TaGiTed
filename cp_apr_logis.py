'''
Compute the nonnegative tensor factorization using alternating Poisson regression
'''
import numpy as np
import time

import ktensor
import sptensor
import tensorTools
from sklearn.linear_model import LogisticRegression

def solveForModeB0(X, M, n, maxInner, epsilon, tol):
    """ 
    Solve for the subproblem B = argmin (B >= 0) f(M)
    
    Parameters
    ----------
    X : the original tensor
    M : the current CP factorization
    n : the mode that we are trying to solve the subproblem for
    epsilon : parameter to avoid dividing by zero
    tol : the convergence tolerance (to 

    Returns
    -------
    M : the updated CP factorization
    Phi : the last Phi(n) value 
    iter : the number of iterations before convergence / maximum
    kktModeViolation : the maximum value of min(B(n), E - Phi(n))
    """
    # Pi(n) = [A(N) kr A(N-1) kr ... A(n+1) kr A(n-1) kr .. A(1)]^T
    Pi = tensorTools.calculatePi(X, M, n)
    #print(M.U[n])
    for iter in range(maxInner):
        # Phi = (X(n) elem-div (B Pi)) Pi^T
        Phi = tensorTools.calculatePhi(X, M.U[n], Pi, n, epsilon=epsilon)
        #print(Phi)
        # check for convergence that min(B(n), E - Phi(n)) = 0 [or close]
        kktModeViolation = np.max(np.abs(np.minimum(M.U[n], 1-Phi).flatten()))
        if (kktModeViolation < tol):
            break
        # Do the multiplicative update
        M.U[n] = np.multiply(M.U[n],Phi)
        #print(" Mode={0}, Inner Iter={1}, KKT violation={2}".format(n, iter, kktModeViolation))
    return M, Phi, iter, kktModeViolation

def solveForModeB1(X, M, n, maxInner, epsilon, tol,sita,Y1, lambta2):
    """
    Solve for the subproblem B = argmin (B >= 0) f(M)

    Parameters
    ----------
    X : the original tensor
    M : the current CP factorization
    n : the mode that we are trying to solve the subproblem for
    epsilon : parameter to avoid dividing by zero
    tol : the convergence tolerance (to
    Returns
    -------
    M : the updated CP factorization
    Phi : the last Phi(n) value
    iter : the number of iterations before convergence / maximum
    kktModeViolation : the maximum value of min(B(n), E - Phi(n))
    DemoU : the updated demographic U matrix
    """
    # Pi(n) = [A(N) kr A(N-1) kr ... A(n+1) kr A(n-1) kr .. A(1)]^T
    Pi = tensorTools.calculatePi(X, M, n)
    #print 'pi='+str(Pi)
    #print(M.U[n])
    for iter in range(maxInner):
        # Phi = (X(n) elem-div (B Pi)) Pi^T
        Phi = tensorTools.calculatePhi(X, M.U[n], Pi, n, epsilon=epsilon)
        #print('phi'+str(Phi))
        #print(Phi)
        # check for convergence that min(B(n), E - Phi(n)) = 0 [or close]
        kktModeViolation = np.max(np.abs(np.minimum(M.U[n], 1-Phi).flatten()))
        if (kktModeViolation < tol):
            break

        #DemoU=__solveDemoU(M.U[n],100,epsilon,Demog,DemoU,lambta1)
        #print 'demoU'
        #print DemoU[0]
        #M.U[n] = np.multiply(M.U[n],Phi)-(lambta1*np.multiply(M.U[n]/(np.sum(Pi,axis=0)),np.dot(np.dot(M.U[n],DemoU)-Demog,DemoU.T)))
        # #M.U[n] = np.multiply(M.U[n],Phi)
        B=M.U[n]
        colNorm = np.apply_along_axis(np.linalg.norm, 0, B, 1)
        zeroNorm = np.where(colNorm == 0)[0]
        colNorm[zeroNorm] = 1
        B = B / colNorm[np.newaxis,  :]
        tm=np.hstack((np.ones((B.shape[0],1)),B))
        Y1=Y1.reshape((Y1.shape[0],1))
        derive=-lambta2*np.dot((Y1/(np.exp(np.multiply(np.dot(tm,sita),Y1))+np.ones((Y1.shape[0],1)))),sita.T)/tm.shape[0]

        M.U[n] = np.multiply(M.U[n],Phi)-(np.multiply(M.U[n],derive[:,1:]))

    return M, Phi, iter, kktModeViolation

def __normalize_mode(M, mode, normtype):
        """Normalize the ith factor using the norm specified by normtype"""
        colNorm = np.apply_along_axis(np.linalg.norm, 0, M.U[mode], normtype)
        zeroNorm = np.where(colNorm == 0)[0]
        colNorm[zeroNorm] = 1
        llmbda = M.lmbda * colNorm
        tempB = M.U[mode] / colNorm[np.newaxis,  :]
        return llmbda,tempB

def __solveSubproblem0(X, M, n, maxInner, isConverged, epsilon, tol):
    """ """
    # Shift the weight from lambda to mode n 
    # B = A(n)*Lambda
    M.redistribute(n)
    # solve the inner problem
    M, Phi, iter, kktModeViolation = solveForModeB0(X, M, n, maxInner, epsilon, tol)
    if (iter > 0):
        isConverged = False
    # Shift weight from mode n back to lambda
    M.normalize_mode(n,1)
    return M, Phi, iter, kktModeViolation, isConverged

def __solveSubproblem1(X, M, n, maxInner, isConverged, epsilon, tol, sita,Y1, lambta2):
    """ """
    # Shift the weight from lambda to mode n
    # B = A(n)*Lambda
    M.redistribute(n)
    # solve the inner problem
    M, Phi, iter, kktModeViolation  = solveForModeB1(X, M, n, maxInner, epsilon, tol,sita,Y1, lambta2)
    if (iter > 0):
        isConverged = False
    # Shift weight from mode n back to lambda
    M.normalize_mode(n,1)
    return M, Phi, iter, kktModeViolation, isConverged

def __solveLogis(B, Y1, maxInner,tol, lambta2,lambta3,sita):
    colNorm = np.apply_along_axis(np.linalg.norm, 0, B, 1)
    zeroNorm = np.where(colNorm == 0)[0]
    colNorm[zeroNorm] = 1
    B = B / colNorm[np.newaxis,  :]
    tm=np.hstack((np.ones((B.shape[0],1)),B))
    Y1=Y1.reshape((Y1.shape[0],1))

    for i in range(maxInner):
        t1=((lambta2*lambta3/tm.shape[1])*sita)
        tt1=np.exp(np.multiply(np.dot(tm,sita),Y1))
        tt2=np.ones((Y1.shape[0],1))
        tt3=np.dot(tm.T,(Y1/(tt1+tt2)))
        t2=(tt3/(tm.shape[0]*lambta2))
        deriv=t1-t2
        if(sum(sum(CItem*CItem for CItem in deriv))<tol):
            break
        sita = sita - 0.001*deriv;
    return sita

    
def cp_apr(X, Y1, R, Minit=None, tol=1e-4, maxiters=1000, maxinner=50,
           epsilon=1e-10, kappatol=1e-10, kappa=1e-2):
    """ 
    Compute nonnegative CP with alternative Poisson regression.
    Code is the python implementation of cp_apr in the MATLAB Tensor Toolbox 
    
    Parameters
    ----------
    X : input tensor of the class tensor or sptensor
    R : the rank of the CP
    lambta1 is the parameter of docomposition of demographic information
    lambta4 is the patameter of penalty item of demoU
    Minit : the initial guess (in the form of a ktensor), if None random guess
    tol : tolerance on the inner KKT violation
    maxiters : maximum number of iterations
    maxinner : maximum number of inner iterations
    epsilon : parameter to avoid dividing by zero
    kappatol : tolerance on complementary slackness
    kappa : offset to fix complementary slackness

    Returns
    -------
    M : the CP model as a ktensor
    cpStats: the statistics for each inner iteration
    modelStats: a dictionary item with the final statistics for this tensor factorization
    """
    N = X.ndims()
     
    ## Random initialization
    if Minit == None:
        F = tensorTools.randomInit(X.shape, R)
        Minit = ktensor.ktensor(np.ones(R), F);
    nInnerIters = np.zeros(maxiters);

    ## Initialize M and Phi for iterations
    M = Minit
    M.normalize(1)
    Phi = [[] for i in range(N)]
    kktModeViolations = np.zeros(N)
    kktViolations = -np.ones(maxiters)
    nViolations = np.zeros(maxiters)

    lambda2=1
    lambda3=1
    sita=np.random.rand(R+1,1);
    ## statistics
    cpStats = np.zeros(7)
    for iteration in range(maxiters):
        startIter = time.time()
        isConverged = True;
        for n in range(N):
            startMode = time.time()
            ## Make adjustments to M[n] entries that violate complementary slackness
            if iteration > 0:
                V = np.logical_and(Phi[n] > 1, M.U[n] < kappatol)
                if np.count_nonzero(V) > 0:
                    nViolations[iteration] = nViolations[iteration] + 1
                    print 'V:',V.shape,V.dtype
                    print 'M.U[n]',M.U[n].shape,M.U[n].dtype
                    M.U[n][V > 0] = M.U[n][V > 0] + kappa
            if n==0:
                sita=__solveLogis(M.U[n],Y1,200,epsilon,lambda2,lambda3,sita)
                M, Phi[n], inner, kktModeViolations[n], isConverged  = __solveSubproblem1(X, M, n, maxinner, isConverged, epsilon, tol,sita,Y1, lambda2)
            else:
                M, Phi[n], inner, kktModeViolations[n], isConverged  = __solveSubproblem0(X, M, n, maxinner, isConverged, epsilon, tol)
            elapsed = time.time() - startMode

        kktViolations[iteration] = np.max(kktModeViolations);
        elapsed = time.time()-startIter
        print("Iteration {0}: Inner Its={1} with KKT violation={2}, nViolations={3}, and elapsed time={4}".format(iteration, nInnerIters[iteration], kktViolations[iteration], nViolations[iteration], elapsed));
        if isConverged:
            break;

    cpStats = np.delete(cpStats, (0), axis=0) # delete the first row which was superfluous
    ### Print the statistics
    #fit = tensorTools.lsqrFit(X,M)
    #ll = tensorTools.loglikelihood(X,[M])
    print("Number of iterations = {0}".format(iteration))
    #print("Final least squares fit = {0}".format(fit))
    #print("Final log-likelihood = {0}".format(ll))
    print("Final KKT Violation = {0}".format(kktViolations[iteration]))
    print("Total inner iterations = {0}".format(np.sum(nInnerIters)))
    
    #modelStats = {"Iters" : iter, "LS" : fit, "LL" : ll, "KKT" : kktViolations[iteration]}
    return M, cpStats