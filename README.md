# svhn-format1

This repository contains the code to preprocess Google's SVHN Format 1 using Python. (This is particularly useful if one does not have access to MatLab and would like each image to consist of the entire number rather than individual digits.

The [publicly available SVHN dataset](http://ufldl.stanford.edu/housenumbers/) comes in two formats. Format 1 contains the full image along with bounding boxes for each digit. Here we determine the box that encloses the entire sequence of digits that makes up the number. We then crop the image and resize it to a 32x32x3 image. This is useful if one were to replicate the paper by [Goodfellow et al](http://arxiv.org/abs/1312.6082), which predicts the sequence of a number along with its digits.
