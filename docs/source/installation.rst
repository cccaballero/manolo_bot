Installation
============

Prerequisites
-------------

* Python 3.12 or higher.
* `uv` (recommended) or `pip`.

Installing from source
----------------------

Clone the repository:

.. code-block:: shell

   git clone https://github.com/cccaballero/manolo_bot.git
   cd manolo_bot

Install dependencies using `uv`:

.. code-block:: shell

   uv sync

Or using `pip`:

.. code-block:: shell

   pip install .

Installing for Development
--------------------------

If you want to contribute to the project or build the documentation locally, install the development dependencies:

.. code-block:: shell

   uv sync --dev

To build the documentation:

.. code-block:: shell

   uv run sphinx-build -b html docs/source docs/_build/html
