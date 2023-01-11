## How to write your own models?

We recommend that you place the code for all your agents in the `my_submisison` directory (though it is not mandatory). We have added random segmentation examples in `random_segmentation_model.py`

**Add your model name in** [`user_config.py`](user_config.py)

## Segmentation model format
You will have to implement a class containing the function `segment_single_image`. This will recieve input image_to_segment, a single frame from onboard the flight. You need to output a 2D image with the pixels values corresponding to the label number.

The labels should be one of the following numbers:

```
class_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255]
```

## What's used by the evaluator
The evaluator uses `MySegmentationModel` from `user_config.py` as its entrypoints. Specify the class name for your model here.

## What's AIcrowd Wrapper

Don't change this file, it is used to save the outputs you predict and read the input images. We run it on the client side for efficiency. The AIcrowdWrapper is the actual class that is called by the evaluator, it also contains some checks to see your predictions are formatted correctly.
