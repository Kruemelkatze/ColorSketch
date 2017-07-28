# ColorSketch
A simple approach to generate inked color sketches from natural images. This happens in three steps:
* **Segmentation**: The image is split into segments using *Mean Shift Segmentation*
* **Inking Mask**: Dark, correlated patches or outlines are extracted from the image.
* **Inking**: All dark patches are "inked" (colored black) in the segmented image.

## Variants
The code includes handles for some basic settings:
* Inking can produce either outlines or patches
* Inking can be done on the original or the segmented image.
* 3 types of grayscale conversions and switchable contrast enhancement (both really impact the end results!)
* Some global fine-tuning parameters and lots of parameters hidden in the code (see them as a kind of easter eggs [the classic, edible ones])

## Example images
Thanks...
* for the **cat** to Wikipedia (CC)
* for **Lenna** to the Image Processing community
* for the **girl** to Maryse Casol (found in the [XDoG-Paper](http://www.kyprianidis.com/p/cag2012/winnemoeller-cag2012.pdf) by Winnemöller et al.)

## Examples
Coming soon :)

## Dependencies
* Python 3 :-)
* OpenCV for Python
* Numpy
* [fjean/pymeanshift](https://github.com/fjean/pymeanshift) - be sure to [install](https://github.com/fjean/pymeanshift/wiki/Install) it!
