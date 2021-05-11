# Singular Value Decomposition to Compress Images

## Files and folders' description

Implementations can be found in `svd.ipynb` along with necessary comments and explanations.

The `input` folder comprises of input images. The `output` folder comprises of output images grouped into subfolders following the name of the input file. 

An example has been run to output `output/sample/images/I<k>.jpg` files, which are the results of compressing `input/sample.jpg` with $K = 10, 20, 30,...$. The compression rates corresponding to the values of `K` are stored in `output/sample/rk.txt`. The two graphs `compression-rate` and `frobenius-error` in `output/sample` illustrate these two quantities with varying values of `K`.

## Comments

From `compression-rate.png` and `rk.txt`, the only acceptable compression rates are those lower than $1$, corresponding to $K \geq 890$. 

In that range, in my opinion, $K = 950$ is the limit for visual sensing, with compression rate about $57.5$ percent. From $K = 960$ and above, the image is highly degraded.