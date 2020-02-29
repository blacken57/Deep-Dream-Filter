from deepdreamer import model, load_image, recursive_optimize
import numpy as np
import PIL.Image
import random

layer_tensor = model.layer_tensors[10]
file_name = "8.jpeg"
img = load_image(filename='{}'.format(file_name))

img_result = recursive_optimize(layer_tensor=layer_tensor, image=img,
                 num_iterations=25, step_size=2, rescale_factor=0.5,
                 num_repeats=8, blend=0.2)

img_result = np.clip(img_result, 0.0, 255.0)
img_result = img_result.astype(np.uint8)
result = PIL.Image.fromarray(img_result, mode='RGB')
result.save(file_name+'_out.jpg')
result.show()
