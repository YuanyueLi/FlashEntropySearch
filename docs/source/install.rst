============
Installation
============


Requirements
============

To use this package, you will need to have the following software and dependencies installed on your system:

- ``Python >= 3.8``

- ``C compiler and Python development headers.``

  - On Linux, you may need to install the ``gcc`` and ``python-dev`` (for apt) or ``python-devel`` (for yum) packages first.
  - On Windows, you will need to install the `Microsoft Visual C++ 14.0 or greater Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_ first.

- ``Numpy >= 1.18``  (will be installed automatically when you install the package from PyPI)

- ``Cython >= 0.29`` (will be installed automatically when you install the package from PyPI)

- ``Cupy >= 8.3.0`` (optional, only required for GPU acceleration)

The ``numpy`` and ``cython`` dependencies will be installed automatically when you install the package from PyPI. The ``cupy`` dependency is optional, and is only required for GPU acceleration. If you want to use GPU acceleration, you need to install ``cupy`` manuall before installing the package from PyPI.

The code is tested on KDE neon 5.27, but it should work on other platforms as well.


From PyPI
============

To install the latest version of the package from PyPI, run the following command:

.. code-block:: bash

  pip install ms_entropy

The installation time should be less than 1 minute, if everything is set up correctly.

From source
============

To install from source, clone the repository and run the following commands:

.. code-block:: bash

  git clone https://github.com/YuanyueLi/FlashEntropySearch.git
  cd FlashEntropySearch
  python setup.py build_ext --inplace

Then you can copy the ``ms_entropy`` folder to your project directory. Or set your ``PYTHONPATH`` environment variable to include the ``ms_entropy`` folder.

Test
====

To test that the package is working correctly, run the example.py script:

.. code-block:: bash

  cd examples
  python example.py
