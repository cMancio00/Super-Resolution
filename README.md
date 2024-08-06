# Super-Resolution

The goal of this project is to reconstruct a high-resolution image from a single low-resolution image.

We will utilize the architecture presented in the paper:
[Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921) (Lim et al. 2017).

# Export the notebook as pdf

```bash
jupyter nbconvert --to pdf notebook.ipynb --output "Model Demonstration" --LatexPreprocessor.title "Super Resolution Demonstration" --LatexPreprocessor.date "August, 2024" --LatexPreprocessor.author_names "Christian Mancini"
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
