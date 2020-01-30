
from skimage import io, transform
from skimage.color import gray2rgb

image = io.imread('data/train/5935a672-23d2-11e8-a6a3-ec086b02610b.jpg')  #problem image
print(io.imshow(image))

image = io.imread('data/train/58767ee4-23d2-11e8-a6a3-ec086b02610b.jpg')
print(io.imshow(image))
