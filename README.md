# BRDF LUT MLP

![](.github/Screenshot%202026-02-22%20at%2014.49.00.png)

A small MLP trained to approximate the split-sum BRDF integration lookup table used in IBL rendering, replacing a precomputed 512×512 texture with a learned function.

## Architecture

6-layer MLP with GELU activations and a Sigmoid output layer. Takes NdotV and roughness as inputs and outputs the scale and bias terms of the split-sum approximation.

| Parameter      | Value                            |
| -------------- | -------------------------------- |
| Inputs         | 2 (NdotV, roughness)             |
| Outputs        | 2 (scale, bias)                  |
| Hidden layers  | 6x32                             |
| Activation     | GELU (hidden), Sigmoid (output)  |
| Loss           | MSE                              |
| Optimizer      | Adam (lr=0.01)                   |

## Results

| Metric                | Value                   |
| --------------------- | ----------------------- |
| Training time         | 115.4s (100 epochs)     |
| Inference (512x512)   | 64.912ms                |
| Model size            | 25.5 KB                 |
| Raw LUT size          | 2048 KB                 |
| Compression ratio     | 79.56x                  |
| Mean FLIP error       | 0.025993                |
