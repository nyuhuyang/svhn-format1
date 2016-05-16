# svhn-format1

This repository contains the code to preprocess Google's SVHN Format 1 using Python. (This is particularly useful if one does not have access to MatLab and would like each image to consist of the entire number rather than individual digits.

The [publicly available SVHN dataset](http://ufldl.stanford.edu/housenumbers/) comes in two formats. Format 1 contains the full image along with bounding boxes for each digit. Here we determine the box that encloses the entire sequence of digits that makes up the number. We then crop the image and resize it to a 32x32x3 image. This is useful if one were to replicate the paper by [Goodfellow et al](http://arxiv.org/abs/1312.6082), which predicts the sequence of a number along with its digits.

The output of `load_data(theano_shared=False)` is a list containing train set, validation set and test set. Each set contains the image data and the labels. So in the following,
```
train_set, validation_set, test_set = load_data(shared=False)
x, y = train_set
```
`x` is a numpy array of flattened 32x32x3 images and `y` are the respective labels. The labels are an array of sequences. Each sequence represents the respectively indexed image in `x`.  The first element of the sequences contains the sequence length (0 = 1 digit number, 1 = 2 digit number, etc.), the second element contains the first digit of the number, and so on.

`src/utils.py` will pickle and compress the SVHN data available online into `trainpkl.gz` and `testpkl.gz`. These files only contain the appropriately cropped and resized original SVHN images. Once these files have been created, load_data(shared=False) will always call `convert_data_format()` to process the data further.
