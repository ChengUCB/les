LES
===

**Latent Ewald Summation** adds long-range electrostatics to a short-range machine learning
interatomic potential, learning it from ordinary energy and force data alone. The network
predicts a latent charge on each atom -- and optionally dipoles, quadrupoles and
polarizabilities -- and LES sums their electrostatics with an Ewald sum.

Nothing supervises those charges. They come out physically meaningful all the same: models
trained on energies and forces predict dipoles, Born effective charges, IR and Raman spectra,
and dielectric response.

`les <https://github.com/ChengUCB/les>`_ is a plug-in library, already integrated into several
MLIPs. Pick yours below; each page shows the training script and explains the options.

.. toctree::
   :maxdepth: 2
   :caption: The method

   theory

.. toctree::
   :maxdepth: 2
   :caption: Using LES

   library
   mace
   cace
   nequip

.. toctree::
   :maxdepth: 1

   citation
