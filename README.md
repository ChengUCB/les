# Latent Ewald Summation (LES)

## Summary

The Latent Ewald Summation (LES) library is a plug-in to add long-range interactions to short-ranged machine learning interatomic potentials.
## Requirements

- Python 3.6 or higher
- NumPy
- PyTorch

## Installation

Please refer to the `setup.py` file for installation instructions.

`les` can be installed using `pip`
```bash
git clone https://github.com/ChengUCB/les.git
pip install -e . 
```

## Usage

We present **LES (Latent Ewald Summation)** ([https://github.com/ChengUCB/les](https://github.com/ChengUCB/les)) as a plug-in library designed to add long-range interactions to short-range machine learning interatomic potentials (MLIPs). 

Here we demonstrate its integration with MLIPs such as **MACE**, **NequIP**, **Allegro**, **CACE**, and **CHGNet**, and provide training scripts and trained models. In particular, we provide **MACELES-OFF** trained on the SPICE dataset. 

Here you can find MLIP packages **with LES implementation** presented in [*A Universal Augmentation Framework for Long-Range Electrostatics in Machine Learning Interatomic Potentials*](https://pubs.acs.org/doi/10.1021/acs.jctc.5c01400).

| Package | Link |
|---------|------|
| **CACE**   | [github.com/BingqingCheng/cace](https://github.com/BingqingCheng/cace) |
| **MACE**   | [github.com/ChengUCB/mace](https://github.com/ChengUCB/mace) |
| **MACE(updated)**   | [github.com/ACEsuit/mace](https://github.com/ACEsuit/mace) |
| **NequIP** | [github.com/ChengUCB/NequIP-LES](https://github.com/ChengUCB/NequIP-LES) |
| **Allegro** | [github.com/ChengUCB/NequIP-LES](https://github.com/ChengUCB/NequIP-LES) |
| **MatGL**  | [github.com/ChengUCB/matgl](https://github.com/ChengUCB/matgl) |


**Example training scripts** for these LES-augmented MLIPs can be found in [https://github.com/ChengUCB/les_fit].

**Hyperparameters selection:** The default parameters (i.e. without setting anything) usually work well. 
One thing that can be changed is 'remove_self_interaction'. Setting 'remove_self_interaction=True' is the default and is the most robust choice.
'remove_self_interaction=False' can sometimes yield a bit better training accuracy, but is less robust when training on finite systems and then extrapolate to periodic systems.

## 📣 Update 
[2025-10] The **`MACELES`** model has been implemented in the main [**MACE** repository](https://github.com/ACEsuit/mace). Example training and evaluation scripts are available in [les_fit](https://github.com/ChengUCB/les_fit/tree/main/MLIPs/MACE-LES-new).

## License

This project is licensed under the CC BY-NC 4.0 License.

## Citation

```text
@article{cheng2025latent,
  title={Latent Ewald summation for machine learning of long-range interactions},
  author={Cheng, Bingqing},
  journal={npj Computational Materials},
  volume={11},
  number={1},
  pages={80},
  year={2025},
  publisher={Nature Publishing Group UK London}
}


@article{King2025Machine,
  title = {Machine Learning of Charges and Long-Range Interactions from Energies and Forces},
  author = {King, Daniel S. and Kim, Dongjin and Zhong, Peichen and Cheng, Bingqing},
  year = 2025,
  journal = {Nature Communications},
  volume = {16},
  number = {1},
  pages = {8763},
  publisher = {Nature Publishing Group}
}



@article{zhong2025machine,
  title={Machine learning interatomic potential can infer electrical response},
  author={Zhong, Peichen and Kim, Dongjin and King, Daniel S and Cheng, Bingqing},
  journal={arXiv preprint arXiv:2504.05169},
  year={2025}
}

@article{Kim2025Universalb,
  title = {A Universal Augmentation Framework for Long-Range Electrostatics in Machine Learning Interatomic Potentials},
  author = {Kim, Dongjin and Wang, Xiaoyu and Vargas, Santiago and Zhong, Peichen and King, Daniel S. and Inizan, Theo Jaffrelot and Cheng, Bingqing},
  year = 2025,
  journal = {Journal of Chemical Theory and Computation},
  publisher = {American Chemical Society},
  doi = {10.1021/acs.jctc.5c01400}
}


```

## Contact

For any queries regarding LES, please contact Bingqing Cheng at tonicbq@gmail.com.
