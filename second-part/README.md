
The [two checkpoints (available here)](https://drive.google.com/drive/folders/1LQoBi7LSg9nArTPMtD2d_83Oi-V_9CZs?usp=sharing) of the model were trained from scratch two different ways. In the first, i took the outputs of err’s model and then went in fiji and thresholded it, then dilated it to fill in some gaps, and then used particle analyzer to remove the noise. I then used a script to do the following 
    1. For each segment folder in a parent folder, open the segment folder, then open the inklabels file, and select a random 1028x1028 region. If that region contains more than 15% ink label (% of pixels that are 255), crop that region, and then crop the same region from the mask png, and from each of the tifs numbered 15-50.tif, then place each of these in a unique folder. Repeat for the number of specified regions.
    2. Once the region number is met, randomly grab 2 of these regions and apply a 50% chance to flip and 50% chance of 90 degree rotation to each individual set of tifs, masks, and labels for each respective region, and then stitch them together horizontally

These steps were performed on segments ending in 5753, 3336, 4423, 51002, and 1321. I did this to create a synthetic dataset to train from, given the scarcity of actual high quality training data. 

The second checkpoint was created by doing a similar but different set of steps after the fiji thresholding and particle analyzing as follows:

    1. For each segment in a parent folder, open the segment folder, and then open the inklabels folder. Use cv2 to create bounding boxes around areas that meet a specified curvature and size. Log this bounding box location to a JSON in coco format, a text file, and a .png with the bounding boxes and place it in a folder called <segmentid>_results
    2. Then, with a separate script, for each segment in a parent folder, open the text file or json file containing bounding box location information, and cut the mask, label, and tif files for those locations and place all of them in a separate folder called “cutout_number”. 

This was started as a bit of a proof-of-concept into creating a MNIST style vesuvius dataset that one simply pull and use for training data.

Both of these approaches yielded significantly better results than any other form of labeling we had performed, and both are thankfully diverse datasets in the context of this challenge, and seem to have caused our model to generalize very well. There's probably a better way to do these without scripts, but they are provided here as well.

Both of these models are just basic forks of youssef’s first letters code, modified to just train or infer on anything in specified folders. For one of the models a slight change is made to the get_train_valid_dataset function, which you can review as well. This was implemented based on a suggestion from other discord member JakeGonder.

Both were trained for very few epochs, because i simply ran out of time. I made these datasets on 12/30 around 0200 US Central time.

If you want to repeat this process, simply grab some preds from the discord or publicly available repository, put them in fiji and threshold away the noise, then go to analyze>analyze particles> and pick a size somewhere around 5,000 or so to infinity, then show masks and hit ok to get the rest of the noise out. Then click edit>invert to turn the mask to the right colors. Then you can run those other scripts on them and end up with a similar dataset from which to train youssef’s model. 
