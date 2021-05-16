|Name|Student ID|Mail|
|---|---|---|
|Vũ Lê Thế Anh|20C13002|anh.vu2020@ict.jvn.edu.vn|

<br />

# Principal Component Analysis to Compress Images/Audio files

## Files and folders' description

Implementation of PCA can be found in [pca.py](pca.py), the scripts to apply it is in [pca_compress.ipynb](pca_compress.ipynb) along with necessary comments and explanations.

The [input](input) folder comprises of input images and wavfiles. The [output/images](output/images) folder comprises of output images grouped into subfolders following the name of the input file. The same is applied for audio files in [output/audios](output/audios).

An example has been run to output [output/sample/images](output/images/sample/imgs)`/I<k>.jpg` files, which are the results of compressing [input/sample.jpg](input/sample.jpg) with `K = 10, 20, 30,...`. The compression rates corresponding to the values of `K` are stored in [rk.txt](output/images/sample/rk.txt). The graph [compression-rate.png](output/images/sample/compression-rate.png) illustrates these two quantities with varying values of `K`.

## Comments

### Sample image

From [compression-rate.png](output/images/sample/compression-rate.png) and [rk.txt](output/images/sample/rk.txt), the only acceptable compression rates are those lower than `1`, corresponding to `K < 160`. 

In that range, in my opinion, `K = 90` is the limit for visual sensing, with compression rate of about `0.59`.

### Sample audio

From [compression-rate.png](output/audios/sample/compression-rate.png) and [rk.txt](output/audios/sample/rk.txt), the only acceptable compression rates are those lower than `1`, corresponding to `K < 260`. 

In that range, in my opinion, `K = 90` is the limit for audio sensing, with compression rate of about `0.35`.