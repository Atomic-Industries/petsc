function port = openport(number)
%
%  port = openport(number)
%  Opens a port to receive matrices from Petsc.
% see closeport and receive
disp('You must build the openport mex file by doing make BOPT=g matlabcodes')