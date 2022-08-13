# Self-supervised Matting-specific Portrait Enhancement and Generation
>We resolve the ill-posed alpha matting problem from a completely different perspective. Given an input portrait image, instead of estimating the corresponding alpha matte, we focus on the other end, to subtly enhance this input so that the alpha matte can be easily estimated by any existing matting models. This is accomplished by exploring the latent space of GAN models. It is demonstrated that interpretable directions can be found in the latent space and they correspond to semantic image transformations. We further explore this property in alpha matting. Particularly, we invert an input portrait into the latent code of StyleGAN, and our aim is to discover whether there is an enhanced version in the latent space which is more compatible with a reference matting model. We optimize multi-scale latent vectors in the latent spaces under four tailored losses, ensuring matting-specificity and subtle modifications on the portrait. We demonstrate that the proposed method can refine real portrait images for arbitrary matting models, boosting the performance of automatic alpha matting by a large margin. In addition, we leverage the generative property of StyleGAN, and propose to generate enhanced portrait data which can be treated as the pseudo GT. It addresses the problem of expensive alpha matte annotation, further augmenting the matting performance of existing models.
## Description



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
