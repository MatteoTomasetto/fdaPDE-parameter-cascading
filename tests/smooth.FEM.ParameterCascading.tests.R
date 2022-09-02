##########################################
############## TEST SCRIPT ###############
##########################################

library(fdaPDE)

###########################
#### Test 1: 2D domain ####
###########################

library(fdaPDE)
rm(list=ls())
graphics.off()

x = seq(0,1, length.out = 11)
y = x
locations = expand.grid(x,y)

mesh = create.mesh.2D(locations, order = 1)
plot(mesh)

FEMbasis=create.FEM.basis(mesh)

# Test function
a1=1
a2=4
z<-function(p){  
  a1*sin(2*pi*p[,1])*cos(2*pi*p[,2])+a2*sin(3*pi*p[,1])}

# Exact solution (pointwise at nodes)
sol_exact=z(mesh$nodes)
image(FEM(sol_exact, FEMbasis))

DatiEsatti=z(locations)
ndati = length(DatiEsatti)

# Add error to simulate data
set.seed(7893475)
ran=range(DatiEsatti)
data = DatiEsatti + rnorm(ndati, mean=0, sd=0.05*abs(ran[2]-ran[1]))

# Set smoothing parameter
lambda= 10^seq(-6,-3,by=0.25)

# Set stationary PDE parameters
K = matrix(c(1,0,0,4), nrow = 2)
b = c(0,0)
c = 0

#### Test 1.1: Diffusion matrix estimate with GCV exact and L-BFGS-B
parameter_cascading = list(diffusion = c('K','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.2: Diffusion matrix estimate with GCV stochastic and L-BFGS-B
parameter_cascading = list(diffusion = c('K','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.3: Diffusion matrix estimate with GCV exact and Gradient Descent
parameter_cascading = list(diffusion = c('K','Gradient'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.4: Diffusion matrix estimate with GCV stochastic and Gradient Descent
parameter_cascading = list(diffusion = c('K','Gradient'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.5: Diffusion matrix estimate with GCV exact and Genetic algorithm
#parameter_cascading = list(diffusion = c('K','Genetic'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.6: Diffusion matrix estimate with GCV stochastic and Genetic algorithm
#parameter_cascading = list(diffusion = c('K','Genetic'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.7: Diffusion direction estimate with GCV exact and L-BFGS-B
parameter_cascading = list(diffusion = c('K_direction','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.8: Diffusion direction estimate with GCV stochastic and L-BFGS-B
parameter_cascading = list(diffusion = c('K_direction','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.9: Diffusion direction estimate with GCV exact and Gradient Descent
parameter_cascading = list(diffusion = c('K_direction','Gradient'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.10: Diffusion direction estimate with GCV stochastic and Gradient Descent
parameter_cascading = list(diffusion = c('K_direction','Gradient'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.11: Diffusion direction estimate with GCV exact and Genetic algorithm
#parameter_cascading = list(diffusion = c('K_direction','Genetic'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.12: Diffusion direction estimate with GCV stochastic and Genetic algorithm
#parameter_cascading = list(diffusion = c('K_direction','Genetic'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.13: Diffusion shape estimate with GCV exact and L-BFGS-B
parameter_cascading = list(diffusion = c('K_eigenval_ratio','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.14: Diffusion shape estimate with GCV stochastic and L-BFGS-B
parameter_cascading = list(diffusion = c('K_eigenval_ratio','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.15: Diffusion shape estimate with GCV exact and Gradient Descent
parameter_cascading = list(diffusion = c('K_eigenval_ratio','Gradient'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.16: Diffusion shape estimate with GCV stochastic and Gradient Descent
parameter_cascading = list(diffusion = c('K_eigenval_ratio','Gradient'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.17: Diffusion shape estimate with GCV exact and Genetic algorithm
#parameter_cascading = list(diffusion = c('K_eigenval_ratio','Genetic'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.18: Diffusion shape estimate with GCV stochastic and Genetic algorithm
#parameter_cascading = list(diffusion = c('K_eigenval_ratio','Genetic'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.19: Advection vector estimate with GCV exact and L-BFGS-B
parameter_cascading = list(advection = c('b','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.20: Advection vector estimate with GCV stochastic and L-BFGS-B
parameter_cascading = list(advection = c('b','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.21: Advection vector estimate with GCV exact and Gradient Descent
parameter_cascading = list(advection = c('b','Gradient'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.22: Advection vector estimate with GCV stochastic and Gradient Descent
parameter_cascading = list(advection = c('b','Gradient'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.23: Advection vector estimate with GCV exact and Genetic algorithm
#parameter_cascading = list(advection = c('b','Genetic'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.24: Advection vector estimate with GCV stochastic and Genetic algorithm
#parameter_cascading = list(advection = c('b','Genetic'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.25: Advection direction estimate with GCV exact and L-BFGS-B
parameter_cascading = list(advection = c('b_direction','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.26: Advection direction estimate with GCV stochastic and L-BFGS-B
parameter_cascading = list(advection = c('b_direction','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.27: Advection direction estimate with GCV exact and Gradient Descent
parameter_cascading = list(advection = c('b_direction','Gradient'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.28: Advection direction estimate with GCV stochastic and Gradient Descent
parameter_cascading = list(advection = c('b_direction','Gradient'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.29: Advection direction estimate with GCV exact and Genetic algorithm
#parameter_cascading = list(advection = c('b_direction','Genetic'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.30: Advection direction estimate with GCV stochastic and Genetic algorithm
#parameter_cascading = list(advection = c('b_direction','Genetic'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.31: Advection module estimate with GCV exact and L-BFGS-B
parameter_cascading = list(advection = c('b_intensity','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.32: Advection module estimate with GCV stochastic and L-BFGS-B
parameter_cascading = list(advection = c('b_intensity','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.33: Advection module estimate with GCV exact and Gradient Descent
parameter_cascading = list(advection = c('b_intensity','Gradient'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.34: Advection module estimate with GCV stochastic and Gradient Descent
parameter_cascading = list(advection = c('b_intensity','Gradient'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.35: Advection module estimate with GCV exact and Genetic algorithm
#parameter_cascading = list(advection = c('b_intensity','Genetic'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.36: Advection module estimate with GCV stochastic and Genetic algorithm
#parameter_cascading = list(advection = c('b_intensity','Genetic'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.37: Reaction coefficient estimate with GCV exact and BFGS
parameter_cascading = list(reaction = c('c','BFGS'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.38: Reaction coefficient estimate with GCV stochastic and BFGS
parameter_cascading = list(reaction = c('c','BFGS'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.39: Reaction coefficient estimate with GCV exact and Conjugate Gradient method
parameter_cascading = list(reaction = c('c','CG'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.40: Reaction coefficient estimate with GCV stochastic and Conjugate Gradient
parameter_cascading = list(reaction = c('c','CG'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.41: Reaction coefficient estimate with GCV exact and Nelder-Mead algorithm
parameter_cascading = list(reaction = c('c','Nelder-Mead'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.42: Reaction coefficient estimate with GCV stochastic and Nelder-Mead algorithm
parameter_cascading = list(reaction = c('c','Nelder-Mead'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.43: Anisotropy intensity estimate with GCV exact and L-BFGS-B algorithm
parameter_cascading = list(anisotropy_intensity = c('anisotropy_intensity','L-BFGS-B'))
PDE_parameters = list(K = K, b = c(0,1), c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.44: Anisotropy intensity estimate with GCV stochastic and L-BFGS-B algorithm
parameter_cascading = list(anisotropy_intensity = c('anisotropy_intensity','L-BFGS-B'))
PDE_parameters = list(K = K, b = c(0,1), c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.45: Anisotropy intensity estimate with GCV exact and Gradient Descent
parameter_cascading = list(anisotropy_intensity = c('anisotropy_intensity','Gradient'))
PDE_parameters = list(K = K, b = c(0,1), c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.46: Anisotropy intensity estimate with GCV stochastic and Gradient Descent
parameter_cascading = list(anisotropy_intensity = c('anisotropy_intensity','Gradient'))
PDE_parameters = list(K = K, b = c(0,1), c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.47: Anisotropy intensity estimate with GCV exact and Genetic algorithm
#parameter_cascading = list(anisotropy_intensity = c('anisotropy_intensity','Genetic'))
#PDE_parameters = list(K = K, b = c(0,1), c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.48: Anisotropy intensity estimate with GCV stochastic and Genetic algorithm
#parameter_cascading = list(anisotropy_intensity = c('anisotropy_intensity','Genetic'))
#PDE_parameters = list(K = K, b = c(0,1), c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
#image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.49: all PDE parameters estimate
parameter_cascading = list(diffusion = c('K','L-BFGS-B'), advection = c('b','L-BFGS-B'), reaction = c('c','BFGS'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis,
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 1.50: all PDE parameters estimate in space-varying case
# Set space-varying PDE parameters (they musy be constant spatial function to be estimated via Parameter Cascading)
K_func<-function(points)
{
  output = array(c(1,0,0,4), c(2, 2, nrow(points)))
  return(output)
}

b_func<-function(points)
{
  output = array(0, c(2, nrow(points)))
  for (i in 1:nrow(points)){
    output[2,i] = 1
  }
  return(output)
}

c_func<-function(points)
{
  output = rep(c(0), nrow(points))
  return(output)
}

u_func<-function(points)
{
  output = rep(c(0), nrow(points))
  return(output)
}

parameter_cascading = list(diffusion = c('K','L-BFGS-B'), advection = c('b','L-BFGS-B'), reaction = c('c','BFGS'))
PDE_parameters = list(K = K_func, b = b_func, c = c_func, u = u_func, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis,
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton_fd', DOF.evaluation='stochastic', lambda.selection.lossfunction='GCV')
image(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

###########################
#### Test 2: 3D domain ####
###########################

library(fdaPDE)
rm(list=ls())
graphics.off()
rm(list=ls())

# Function to generate random points in a sphere
rsphere <- function(n, r = 1.0, surface_only = FALSE) {
  phi       <- runif(n, 0.0, 2.0 * pi)
  cos_theta <- runif(n, -1.0, 1.0)
  sin_theta <- sqrt((1.0-cos_theta)*(1.0+cos_theta))
  radius <- r
  if (surface_only == FALSE) {
    radius <- r * runif(n, 0.0, 1.0)^(1.0/3.0)
  }
  
  x <- radius * sin_theta * cos(phi)
  y <- radius * sin_theta * sin(phi)
  z <- radius * cos_theta
  
  cbind(x, y, z)
}


# Build mesh: Sphere
data("sphere3Ddata")
mesh_sphere<-create.mesh.3D(sphere3Ddata$nodes,
                            sphere3Ddata$tetrahedrons,
                            order=1)

FEMbasis <- create.FEM.basis(mesh_sphere)

set.seed(5847947)

# Exact test function
nnodes = nrow(mesh_sphere$nodes)

# Set PDE parameters (in this case they are constant)
K = diag(c(1,.5,1))
b = c(0,0,0)
c = -4*pi^2

# Evaluate exact solution on mesh nodes
exact_sol =  sin(2*pi*mesh_sphere$nodes[,1]) +  2 * sin(2*pi*mesh_sphere$nodes[,2]) +  sin(2*pi*mesh_sphere$nodes[,3])

# Plot exact solution
plot(FEM(exact_sol,FEMbasis))

# Add noise to generate data - 10% level of noise
data=exact_sol + rnorm(nrow(mesh_sphere$nodes), mean=0, sd=0.10*diff(range(exact_sol)))

#### Test 2.1: Diffusion matrix estimate with GCV exact and L-BFGS-B
#parameter_cascading = list(diffusion = c('K','L-BFGS-B'))
#PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
#output_CPP<-smooth.FEM(observations=data, 
#                       FEMbasis=FEMbasis, 
#                       PDE_parameters=PDE_parameters,
#                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
#plot(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 2.2: Advection vector estimate with GCV exact and L-BFGS-B
parameter_cascading = list(advection = c('b_intensity','L-BFGS-B'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
plot(FEM(output_CPP$fit.FEM$coeff,FEMbasis))

#### Test 2.3: Reaction coefficient estimate with GCV exact and BFGS
parameter_cascading = list(reaction = c('c','BFGS'))
PDE_parameters = list(K = K, b = b, c = c, parameter_cascading = parameter_cascading)
output_CPP<-smooth.FEM(observations=data, 
                       FEMbasis=FEMbasis, 
                       PDE_parameters=PDE_parameters,
                       lambda.selection.criterion='newton', DOF.evaluation='exact', lambda.selection.lossfunction='GCV')
plot(FEM(output_CPP$fit.FEM$coeff,FEMbasis))
