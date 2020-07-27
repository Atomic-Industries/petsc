==============================
Developing PETSc Documentation
==============================

.. toctree::
   :maxdepth: 2


General Guidelines
==================

* Good documentation should be like a bonsai tree: alive, on display, frequently tended, and as small as possible (adapted from `these best practices <https://github.com/google/styleguide/blob/gh-pages/docguide/best_practices.md>`__).
* Wrong, irrelevant, or confusing documentation is worse than no documentation.

.. _docs_build:

Building Main Documentation
===========================

The documentation tools listed below (except for pdflatex) are
automatically downloaded and installed by ``./configure``.

* `Sowing <http://ftp.mcs.anl.gov/pub/sowing/sowing.tar.gz>`__: a text processing tool developed by Bill Gropp.  This produces the PETSc manual pages; see the `Sowing documentation <http://wgropp.cs.illinois.edu/projects/software/sowing/doctext/doctext.htm>`__ and :ref:`manual_page_format`.
* `C2html <http://ftp.mcs.anl.gov/pub/petsc/c2html.tar.gz>`__: A text processing package. This generates the HTML versions of all the source code.
* A version of pdflatex, for example in  `Tex Live <http://www.tug.org/texlive/>`__.  This package might already be installed on most systems. It is required to generate the users manual (part of the PETSc documentation).

Note: Sowing and c2html have additional dependencies like gcc, g++, and flex and do not
use compilers specified to PETSc configure. [Windows users please install the corresponding
cygwin packages]

Once pdflatex is in your ``PATH``, you can build the documentation with:

.. code-block:: bash

    make alldoc LOC=${PETSC_DIR}

(Note that this does not include :ref:`sphinx_documentation`).

To get a quick preview of manual pages from a single source directory (mainly to debug the manual page syntax):

.. code-block:: bash

    cd $PETSC_DIR/src/snes/interface
    make LOC=$PETSC_DIR manualpages_buildcite
    browse $PETSC_DIR/docs/manualpages/SNES/SNESCreate.html  # or suitable command to open the HTML page in a browser

.. _sphinx_documentation:

Sphinx Documentation
====================

The Sphinx documentation is currently not integrated into the main docs build as described
in :ref:`docs_build`.

`ReadTheDocs <readthedocs.org>`__ generates the documentation at
https://docs.petsc.org from the `PETSc Git repository <https://gitlab.com/petsc/petsc>`__.

Building the Sphinx docs locally
--------------------------------

* Make sure that you have a recent version of Python 3 and the required modules, as listed in the `ReadTheDocs configuration file <https://github.com/petsc/petsc/blob/master/.readthedocs.yml>`__ and `requirements file for ReadTheDocs <https://github.com/petsc/petsc/blob/master/src/docs/sphinx_docs/requirements.txt>`__ (we use a precise version of Sphinx to avoid issues with our custom extension to create inline links).
* Navigate to the location of ``conf.py`` for the Sphinx docs (currently ``src/docs/sphinx_docs``).
* ``make html``
* Open ``_build/html/index.html`` with your browser.

.. _sphinx_guidelines:

Sphinx Documentation Guidelines
-------------------------------

* Use the ``includeliteral`` directive to directly include pieces of source code, as in
  the following example. Note that an "absolute" path has been used, which means
  relative to the root for the Sphinx docs (where ``conf.py`` is found).

.. code-block:: rst

    .. literalinclude:: /../../../src/sys/error/err.c
       :language: c
       :start-at: PetscErrorCode PetscError(
       :end-at: PetscFunctionReturn(0)
       :append: }

* We use the `sphinxcontrib-biblatex extension <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`__.
  This does not currently work perfectly, but its `development branch <https://github.com/mcmtroffaes/sphinxcontrib-bibtex>`__
  promises to support our use case, so we're delaying to see if it's ever released.

Porting LaTeX to Sphinx
-----------------------

These are instructions relevant to porting the Users manual from its previous
LaTeX incarnation, to Sphinx (as here). This section can be removed once the
manual and TAO manual are ported.

The first steps are to modify the LaTeX source to the point that it can
be converted to RST by `Pandoc <pandoc.org>`__.

* Copy the target file, say ``cp manual.tex manual_consolidated.tex``
* copy all files used with ``\input`` into place, using e.g. ``part1.tex`` instead of ``part1tmp.tex`` (as we don't need the HTML links)
* Remove essentially all of the preamble, leaving only ``\documentclass{book}`` followed by ``\begin{document}``
* Save a copy of this file, say ``manual_to_process.tex``.
* Perform some global cleanup operations, as with this script

  .. code-block:: bash

      #!/usr/bin/env bash

      target=${1:-manual_to_process.tex}
      sed=gsed  # change this to sed on a GNU/Linux system

      # \trl{foo} --> \verb|foo|
      # \lstinline{foo} --> \lstinline|foo|
      # only works if there are no }'s inside, so we take care of special cases beforehand,
      # of the form \trl{${PETSC_DIR}/${PETSC_ARCH}/bar/baz} ane \trl{${FOO}/bar/baz}

      ${sed} -i 's/\\trl{${PETSC_DIR}\/${PETSC_ARCH}\([^}]*\)}/\\verb|${PETSC_DIR}\/${PETSC_ARCH}\1|/g' ${target}
      ${sed} -i 's/\\trl{${\([^}]*\)}\([^}]*\)}/\\verb|${\1}\2|/g' ${target}

      ${sed} -i       's/\\trl{\([^}]*\)}/\\verb|\1|/g' ${target}
      ${sed} -i 's/\\lstinline{\([^}]*\)}/\\verb|\1|/g' ${target}

      ${sed} -i 's/\\lstinline|/\\verb|/g' ${target}

      ${sed} -i 's/tightitemize/itemize/g' ${target}
      ${sed} -i 's/tightenumerate/enumerate/g' ${target}

      ${sed} -i 's/lstlisting/verbatim/g' ${target}
      ${sed} -i 's/bashlisting/verbatim/g' ${target}
      ${sed} -i 's/makelisting/verbatim/g' ${target}
      ${sed} -i 's/outputlisting/verbatim/g' ${target}
      ${sed} -i 's/pythonlisting/verbatim/g' ${target}

* Fix any typos like this (extra right brace) : ``PetscViewerPushFormat(viewer,PETSC_VIEWER_BINARY_MATLAB}``
  These will produce very unhelpful Pandoc error messages at the end of the file like
  ``Error at "source" (line 4873, column 10): unexpected end of input %%% End:``
* Convert to ``.rst`` with pandoc (tested with v2.9.2), e.g. ``pandoc -s -t rst -f latex manual_to_process.tex -o manual.rst``.
* Move to Sphinx docs tree (perhaps renaming or splitting up) and build.

Next, one must examine the output, ideally comparing to the original rendered LaTeX, and make fixes on the ``.rst`` file, including but not limited to:

* Check links
* Add correct code block languages when not C, e.g. replace ``::`` with ``.. code-block:: bash``
* Re-add citations with ``:cite:`` (see examples in the dev manual)
* Fix footnotes
* Fix section labels and links
* Fix links with literals in the link text
* Itemized lists
* Replace Tikz with graphviz (or images or something else)
* Replace/fix tables
* Replace included source code with "literalinclude" (see :ref:`sphinx_guidelines`)
* (please add more common fixes here as you find them) ...
