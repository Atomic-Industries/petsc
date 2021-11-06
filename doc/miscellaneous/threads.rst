.. _doc_threads:

=================
Threads and PETSc
=================

Though the current and planned programming model for PETSc is pure
MPI + GPUs (:`_doc_gpus`) we do provide some support for the
use of a hybrid MPI-thread model. Hybrid approaches can be developed in many ways that
affect usability and performance and this section discusses PETSc's
support for threads, which amounts to OpenMP non-offload (refer to
:`_doc_gpus` for any discussion of OpenMP offload).


The simple model of PETSc with threads: back-ends
=================================================

One may contain all the thread operations inside the Mat and Vec
classes, or a 3rd party solver library like hypre or SuperLU, leaving the
user's programming model identical to what it is today. This is
identical to our support model for GPUs :`_doc_gpus`.

A simple model of PETSc with threads: serial PETSc object in a thread
==================================================================================

On my have individual threads (OpenMP or others) manage their own
(sequential) PETSc objects (and each thread can interact only with its
own objects). This is useful when one has many small systems (or sets
of ODEs) that must be integrated in an "embarrassingly parallel"
fashion. Thread safety in PETSc amount to supporting this model with
OpenMP.

To use this feature one must ``configure`` PETSc with the option
``--with-threadsafety --with-log=0 [--with-openmp or
--download-concurrencykit]``. ``$PETSC_DIR/src/dm/impls/swarm/tests/ex7.c``
and ``$PETSC_DIR/src/ksp/ksp/tutorials/ex61f.F90`` demonstrate
how this may be used with OpenMP. The code uses a small number of ``#pragma omp critical``
in non-time-critical locations in the code.

A label, after the ``Collective`` label, for thread safety is added to
the man page for a method that has been tested. An example is in
`segbuffer.c <../../src/sys/utils/segbuffer.c.html>`__, the
``PetscSegBufferGet`` and ``PetscSegBufferUnuse`` methods.

OpenMP examples
---------------

The thread safety in `segbuffer.c
<../../src/sys/utils/segbuffer.c.html>`__  enables the use of seval
``DMPlex`` and ``DMSwarm`` methods, created with PETSC_COMM_SELF, in
an OpenMP thread loop as demonstrated in
``$PETSC_DIR/src/dm/impls/swarm/tests/ex7.c``. For example

..
.. literalinclude:: /../src/dm/impls/swarm/tests/ex7.c
   :start-at: PetscErrorCode createSwarm(
   :end-at: }

After the first assembly of an CPU ``Mat`` the ``MatSetValues()``
method should be thread safe, however the user must insure that there
is no contention with threads, with coloring, for example.

.. note::

   PETSc is not *generically* thread-safe

   All the PETSc objects created during a simulation do not have locks associated with
   them. Select methods have been protected with OpenMP ``critical``
   sections and the configure options above only provide some
   necessary support for thread safety. PETSc practices object
   oriented programming and this goes a long way in supporting
   threads. In fact, fixes to support threads are for the most part
   required where PETSc fails to follow object oriented principles.

Calling non thread safe methods from threads
============================================

To call a non-threadsafe PETSc method in a thread parallel region, one
must synchronize before calling a non-threadsafe function, only one
thread may call it at a time and a memory fence is needed after
mutation by one thread before changes can be observed in-order by
another thread.


Some concerns about a thread model for parallelism
==================================================

A thread model for a parallel library posses some challenges that an
application does not face. For example, there is no mechanism that we
know of for a library, or any subroutine, to know if it is in a thread
and if there are any threads available. For example, one can use a serial
solver using the third party library SuperLU and one should be able to
call this serial SuperLU inside of an OpenMP thread. However, SuperLU
uses OpenMP by default and care must be taken to turn that feature
off in a thread safe build, otherwise oversubscription of threads
could result in poor performance or a program crash.

Additionally care should be taken in using threads as threaded codes
are notoriously difficult to debug.

.. seealso::

   The Problem with Threads, Edward A. Lee, Technical Report No. UCB/EECS-2006-1 January
   10, 2006

