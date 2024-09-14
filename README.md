# Super-Resolution

The goal of this project is to reconstruct a high-resolution image from a single low-resolution image.

We will utilize the architecture presented in the paper:
[Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921) (Lim et al. 2017).

# Project structure

- [checkpoint](/checkpoint/) Here you will find the state of the model.
> [!NOTE]
> `SR_c64_rb8_e50_202408051714.pth` indicates that the SuperResolution model has 64 channels, 8 Residual Blocks, 
> and has been trained for 50 epochs. This represents the state of the best model during the model selection phase.
> The other numbers are simply the timestamp of when the model was saved.
- **data** This folder will be automatically created when you start the project. The dataset will be downloaded here.
- [dataset](/dataset/) This package contains the module [data_preparation](/dataset/data_preparation.py) for downloading and splitting the dataset,
as well as the module [super_resolution_dataset](/dataset/super_resolution_dataset.py), which contains the extended **Dataset** class used for the model.
- [SRM](/SRM/) This package contains the module [modules](/SRM/modules.py), which includes the building block layers
(**ResidualBlocks** and **Upsample**) for the module [network](/SRM/network.py), where the SuperResolutionNetwork is defined.
- [utils](/utils/) This package contains some useful methods for training and the [model_selection](/utils/training_utilitis.py#L129) method.
- **output** This folder contains the output images obtained during validation and testing.
- [training_logs](/training_logs/) contains the CSV files of loss and PSNR for the training, both before and after validation of the best model.
- [notebook](/notebook.ipynb) This notebook displays the main results of the model.
- [main](/main.py) This is what needs to be run to perform all the tasks, from downloading the dataset to testing the model. A seed has been set, i.e., `777`, to ensure consistent results.

# Run the code
Once the steps described in [Installation of Requirements and Kernel](#installation-of-requirements-and-kernel)
is completed, you can run the code with the following command

```bash
python3 main.py
```
> [!WARNING]
> If you have an Nvidia card on Xorg and you have suspended the pc it is
> common that the GPU will not turn on again properly and will appear busy, 
> so unusable for computation.

# Export the notebook as pdf

```bash
jupyter nbconvert --to pdf notebook.ipynb --output "ModelDemonstration" --LatexPreprocessor.title "Super Resolution Demonstration" --LatexPreprocessor.date "September, 2024" --LatexPreprocessor.author_names "Christian Mancini"
```

# Installation of Requirements and Kernel

In the project directory, execute the following commands:

```bash
python3 -m venv .venv
```
> [!NOTE]
> The name of the virtual environment will match the name of the hidden folder, 
> in this case, `.venv`.

To activate the virtual environment, run:

```bash
source .venv/bin/activate
```
Next, install the required packages with:

```bash
pip install --upgrade pip & pip install -r requirements.txt
```

Now, we need to set up the virtual environment as a Jupyter kernel:

```bash
python -Xfrozen_modules=off -m ipykernel install --user --name=super-resolution
```
You can now select `super-resolution` as your kernel.

To view the installed kernels, use:

```bash
jupyter kernelspec list
```
The output should resemble the following:

```
Available kernels:
  python3      /home/mancio/PycharmProjects/super-resolution/.venv/share/jupyter/kernels/python3
  super-resolution    /home/mancio/.local/share/jupyter/kernels/super-resolution
```
> [!TIP]  
> To remove a kernel, you can use the following command:

```bash
jupyter kernelspec uninstall super-resolution -y
```
