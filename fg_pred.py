from pymatting import *
import numpy as np

scale = 1.0

image = load_image("./image/00130.png", "RGB", scale, "box")
trimap = load_image("./trimap/00130.png", "GRAY", scale, "nearest")

# estimate alpha from image and trimap
alpha = estimate_alpha_cf(image, trimap)

# make gray background
background = np.zeros(image.shape)
background[:, :] = [0.5, 0.5, 0.5]

# estimate foreground from image and alpha
foreground = estimate_foreground_ml(image, alpha)

save_image("foreground/00130.png", foreground)
# # blend foreground with background and alpha, less color bleeding
# new_image = blend(foreground, background, alpha)

# # save results in a grid
# images = [image, trimap, alpha, new_image]
# grid = make_grid(images)
# save_image("lemur_grid.png", grid)

# # save cutout
# cutout = stack_images(foreground, alpha)
# save_image("lemur_cutout.png", cutout)

# # just blending the image with alpha results in color bleeding
# color_bleeding = blend(image, background, alpha)
# grid = make_grid([color_bleeding, new_image])
# save_image("lemur_color_bleeding.png", grid)
