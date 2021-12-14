# import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import RandomRotation
from prepare_data import prepare_data


def gasuss_noise(image, standard_diviation):
    noise_tensor = np.zeros(image.shape)
    noise = np.random.normal(0, standard_diviation, (28, 28, 1))
    noise_tensor[1:3] = noise
    out = image + noise_tensor
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)

    return out


# load test data
(test_image_orig, test_label_orig) = prepare_data()[1]

# rotation
test_image_rot = RandomRotation(1/8)(test_image_orig)
test_label_rot = test_label_orig

# add gaussian noise
test_image_noise = gasuss_noise(test_image_orig, 3)
test_label_noise = test_label_orig

# for i in range(5):
#     plt.subplot(5, 1, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     # plt.imshow(test_image_orig[i], cmap=plt.cm.binary)
#     # plt.xlabel(test_label_orig[i])
#     plt.imshow(test_image_noise[10+i])
#     plt.xlabel(test_label_rot[10+i])
# plt.show()
