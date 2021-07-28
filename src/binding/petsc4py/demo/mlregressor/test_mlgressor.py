import petsc4py, sys
petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy as np

comm = PETSc.COMM_WORLD
size = comm.getSize()
rank = comm.getRank()

OptDB = PETSc.Options()

y_array = [0.2, -0.3, -0.8, -0.3, 1.2]
X_array = [-1.0, 0.5, -0.5, -0.25, 0.0, -0.5, 0.5, -0.25, 1.0, 0.5]
rows_ix = [0, 1, 2, 3, 4]
cols_ix = [0, 1]

X = PETSc.Mat().create(comm=comm)
X.setSizes((5,2))
X.setFromOptions()
X.setUp()

y = PETSc.Vec().create(comm=comm)
y.setSizes(5)
y.setFromOptions()

if not rank :
    X.setValues(rows_ix,cols_ix,X_array,addv=True)
    y.setValues(rows_ix,y_array,addv=False)

X.assemblyBegin(X.AssemblyType.FINAL)
X.assemblyEnd(X.AssemblyType.FINAL)
y.assemblyBegin()
y.assemblyEnd()

y_predicted = y.duplicate()

mlregressor = PETSc.MLRegressor().create(comm=comm)
mlregressor.setType(PETSc.MLRegressor.Type.LINEAR)
mlregressor.fit(X,y)
mlregressor.predict(X,y_predicted)
y_predicted.view()



