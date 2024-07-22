# Super-Resolution

The aim image a super-resolution is to reconstruct a high-resolution image from a single low resolution image.

We will use the architecture presentend in:
[Enhanced Deep Residual Networks for Single Image Super-Resolution](https://arxiv.org/pdf/1707.02921) (Lim et all 2017).

# Install the requirements and the kernel

In the project folder run the following commands:

```bash
python3 -m venv .venv
```
> [!NOTE]
> The name of the virtual environment will be the same as the name of hidden folder, 
>in this case `.venv`.

The virtual environment can be activated with:

```bash
source .venv/bin/activate
```
The requirements can be installed with:

```bash
pip install --upgrade pip & pip install -r requirements.txt
```

We just now need to make the virtual environment a Jupyter kernel.

```bash
python -Xfrozen_modules=off -m ipykernel install --user --name=super-resolution
```
Now you can choose `super-resolution` as a Kernel.

We can see the installed kernels with:

```bash
jupyter kernelspec list
```
The output should be something like this:

```
Available kernels:
  python3      /home/mancio/PycharmProjects/super-resolution/.venv/share/jupyter/kernels/python3
  super-resolution    /home/mancio/.local/share/jupyter/kernels/super-resolution
```
> [!NOTE]  
> You can remove a kernel with the following command:

```bash
jupyter kernelspec uninstall super-resolution -y
```
