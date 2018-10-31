"""/*
 * Copyright (C) 2018 Learning Algorithms and Systems Laboratory, EPFL, Switzerland
 * Author:  Lukas Huber and Sina Mirrazavi and Nadia Figueroa
 * email:   {lukas.huber,sina.mirrazavi,nadia.figueroafernandez}@epfl.ch
 * website: lasa.epfl.ch
 *
 * This work was supported by the EU project Cogimon H2020-ICT-23-2014.
 *
 * Permission is granted to copy, distribute, and/or modify this program
 * under the terms of the GNU General Public License, version 2 or any
 * later version published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details
 */
"""

# #include "lpvDS.h"
import numpy as np
import numpy.linalg as LA

import yaml

class lpvDS:
    def __init__(self, K=[], M=[], Priors=[], Mu=[], Sigma=[],  A=[], attractor=[], name='default', fileName = 0):
        #/* Given the parameters directly as MatrixXd, initialize all matrices*/
        # && /* Given the parameters directly as vector<double>, initialize all matrices*/

        if isinstance(fileName, str):
            ds_gmm_dict=yaml.load(open(fileName))
            K = ds_gmm_dict['K']
            M = ds_gmm_dict['M']
            Priors =ds_gmm_dict['Priors']
            Mu = ds_gmm_dict['Mu']
            Sigma = ds_gmm_dict['Sigma']
            A = ds_gmm_dict['A']
            attractor = ds_gmm_dict['attractor']
        
        self.K = int(K)
        self.M = int(M)
        
        self.attr = attractor
        self.gamma   = np.zeros((self.K))


        if np.array(A).shape[0]> self.K: # input as one vector
            # Transform vector to matrix
            # .swapaxes() is used to align with column/row convention
            self.A_Matrix= np.reshape(A, (self.K, self.M, self.M) ).swapaxes(1,2)
            self.Prior   = np.array(Priors)
            self.Mu      = np.reshape(Mu, (self.K, self.M) )
            self.Sigma   = np.reshape(Sigma, (self.K, self.M, self.M) ).swapaxes(1,2)
            
        else: # Input vector already has got the right shape
            self.A_Matrix = np.array(A)
            self.Prior    = np.array(Priors)
            self.Mu       = np.array(Mu)
            self.Sigma    = np.array(Sigma)
            self.gamma    = np.zeros((self.K))

        # Check dimensions of vectors -- TODO for general shape
        if (np.array(Priors).shape[0] != self.K):
            print("Initialization of Prior is wrong.")
            print("Number of components is: {}".format(self.K) )
            print("Dimension of states of Prior is: {}".format(Priors.size()) )
            #ERROR()
        if (np.array(Mu).shape[0] != self.K*self.M):
            print("Initialization of Sigma-matrices is wrong.")
            print("Size of vector should be K ({})*M({})={}".format(self.K, self.M, self.M*self.K) )
            print("Given vector is of size {}".format(np.array(A).shape[0]) ) 
            #ERROR()
        if (np.array(Sigma).shape[0] != self.K*self.M*self.M):
            print("Initialization of Sigma-matrices is wrong.")
            print("Size of vector should be K ({})*M({})*M({})={}".format(self.K, self.M, self.M, self.M*self.M*self.K) )
            print("Given vector is of size {}".format(np.array(A).shape[0]) ) 
            #ERROR()
        if (np.array(A).shape[0] != self.K*self.M*self.M):
            print("Initialization of A-matrices is wrong.")
1            print("Size of vector should be K ({})*M({})*M({})={}".format(self.K, self.M, self.M, self.M*self.M*self.K) )
            print("Given vector is of size {}".format(np.array(A).shape[0]) ) 
            #ERROR()

        print("Initialized an M: {} dimensional GMM-Basedn LPV_DS with K: {} components".format(M, K))


#/****************************************/
#/*     Actual computation functions     */
#/****************************************/
    def compute_A(self, X):
        # /* Calculating the weighted sum of A matrices */
        if ((np.array(X).shape[0] != self.M)):
            print("The dimension of X in compute_A is wrong.")
            print("Dimension of states is: {}".format(self.M) )
            print("Dimension of X {}".format(X.shape[0]) )
	    #ERROR()

        A = np.zeros((self.M,self.M))
        if (self.K>1):
            gamma_= self.compute_gamma(X)
        else:
            gamma_[self.K-1]=1

        for i in range(self.K):
            A = A + self.A_Matrix[i]*gamma_[i]

        return A

    def compute_gamma(self, X):
        gamma = np.zeros(self.K)

        for i in range(self.K):
            gamma[i]=self.Prior[i]*self.GaussianPDF(X,self.Mu[i],self.Sigma[i])

        sum_gamma = np.sum(gamma)
        if (sum_gamma<1e-100):
            for i in range(self.K):
                gamma[i]=1.0/self.K
            else:
                gamma = gamma/sum_gamma

        return gamma

    def compute_f(self, xi, att):
        A_matrix = np.zeros((self.M, self.M))
        xi_dot = np.zeros((self.M))

        A_matrix = self.compute_A(xi)
        xi_dot = A_matrix@(xi - att)
        
        return xi_dot


    def compute_f_check(self, xi, att):
        #/* Check size of input vectors */
        # TODO -- ? OBOSLETE? >> what for

        if (np.array(xi).shape() != self.M):
            print("The dimension of X in compute_f is wrong.")
            print("Dimension of states is: {}".format(self.M) )
            print("You provided a vector of size {}".format(xi.Size() ) ) 
            #ERROR()

        if (np.array(att).shape() != self.M):
            print("The dimension of att in compute_f is wrong.")
            print("Dimension of states is: {}".format(self.M ) ) 
            print("You provided a vector of size {}".format(att.Size() ) ) 
            #ERROR()

        #/* Fill in VectorXd versions of xi and att */
        xi_ = np.zeros((self.M))
        att_ = np.zeros((self.M))
        for m in range(self.M):
            xi_[m]  = xi[m]
            att_[m] = att[m]

        # /* Compute Desired Velocity */
        xd_dot_ = np.zeros((self.M)) 
        xi_dot_ = self.compute_f(xi_,att_)

        # /* Transform Desired Velocity to MathLib form */
        xi_dot = np.zeros((self.M))
        for m in range(self.M):
            xi_dot[m] = xi_dot_[m]

        return xi_dot

    def GaussianPDF(self, x, Mu, Sigma):
        p = []
        gfDiff = np.zeros((1,self.M))
        gfDiff_T = np.zeros((self.M,1))
        SigmaIIInv = np.zeros((self.M, self.M))
        detSigmaII = 0
        gfDiffp = np.zeros((1,1))

        detSigmaII= LA.det(Sigma)
        if (detSigmaII<0):
            detSigmaII=0

        try:
            SigmaIIInv= LA.inv(Sigma)
        except:
            print('WARNING // Sigma NOT invertible')
            pass 

        gfDiff=(x - Mu).transpose()
        gfDiff_T=x - Mu
        gfDiffp =gfDiff*SigmaIIInv* gfDiff_T
        gfDiffp[0,0]=np.abs(0.5*gfDiffp[0,0])
        p = np.exp(-gfDiffp[0,0]) / np.sqrt(pow(2.0*np.pi, self.M)*( detSigmaII +1e-50))

        return p
