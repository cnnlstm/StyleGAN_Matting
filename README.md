# Self-supervised Matting-specific Portrait Enhancement and Generation
>We resolve the ill-posed alpha matting problem from a completely different perspective. Given an input portrait image, instead of estimating the corresponding alpha matte, we focus on the other end, to subtly enhance this input so that the alpha matte can be easily estimated by any existing matting models. This is accomplished by exploring the latent space of GAN models. It is demonstrated that interpretable directions can be found in the latent space and they correspond to semantic image transformations. We further explore this property in alpha matting. Particularly, we invert an input portrait into the latent code of StyleGAN, and our aim is to discover whether there is an enhanced version in the latent space which is more compatible with a reference matting model. We optimize multi-scale latent vectors in the latent spaces under four tailored losses, ensuring matting-specificity and subtle modifications on the portrait. We demonstrate that the proposed method can refine real portrait images for arbitrary matting models, boosting the performance of automatic alpha matting by a large margin. In addition, we leverage the generative property of StyleGAN, and propose to generate enhanced portrait data which can be treated as the pseudo GT. It addresses the problem of expensive alpha matte annotation, further augmenting the matting performance of existing models.
## Description

We present the training code and 5 images for the quick start. 



**Build the environment**
Anaconda is required. 
```
conda env create -f sg_matting.yaml
```
**Download checkpoints**
The pre-trained StyleGAN and matting model checkpoint can be download from [here](https://drive.google.com/uc?id=1h6vVnlFpWk7G2dlzc9DZuKzUuqvloA25). After download the checkpoints, unzip it and move it using:


```
mv ckpt/deeplab_model_best.pth.tar  deeplab_trimap/checkpoint/
```

```
mv ckpt/stylegan2-ffhq-config-f.pt  ./
```

```
mv ckpt/gca-dist-all-data.pth  gca_matting/checkpoints_finetune/
```
**Run the training code on given images**
```
bash train.sh
```


$\color{#FF0000}{Note:}$ 
For training on new image, you need to get the latent codes using the ```stylegan++.py```, besides, you also need to get the trimap using our pre-trained trimap model in(```main.py```) and foreground image (```fg_pred.py```), and put them at the ```dataset/testing```.



## Citation
```
@ARTICLE{9849440,
  author={Xu, Yangyang and Zhou, Zeyang and He, Shengfeng},
  journal={IEEE Transactions on Image Processing}, 
  title={Self-supervised Matting-specific Portrait Enhancement and Generation}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIP.2022.3194711}}
```
