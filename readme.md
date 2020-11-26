Simple python script to read Imagenet64 batch data and write out individual images to their own seperate file.

One reason for this is to allow for more exploration in the image preprocessing steps, batch size etc., as this is no longer hard coded into the files.

Example usage might be to create a pytorch Dataset class that reads the individual images and applies transformations/compositions as the images are requested.

https://github.com/PatrykChrabaszcz/Imagenet32_Scripts

https://patrykchrabaszcz.github.io/Imagenet32/

http://image-net.org/download-images