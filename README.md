# flask-pytorch-inference

This is a sample flask application that uses Bootstrap for the Frontend UI and PyTorch for model inference.

## Create conda environment

```conda env create -f environment.yml```

## How to Use: Resnet50

```bash
activate pytorch_inference
python app_resnet50.py
```

Go to 127.0.0.1:4555 on your browser.

## How to Use: Resnet50 (Transfer Learning)

```bash
activate pytorch_inference
python app_resnet50_c10.py
```

Go to 127.0.0.1:4556 on your browser.


## Output

You should get a screen shown below.

Choose Browse and upload an image. Click Upload to get the output

![](imgs/giraffe-1330814_640.jpg)


![](imgs/out_giraffe-1330814_640.jpg)
