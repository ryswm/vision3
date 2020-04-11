fprintf("Part 1 Our Approach:\n");
fprintf("In order to populate the nonfaces image set, a random image was chosen, as well as a random row and column (With the constraints of being 36 pixels from the right hand border of the image)\n");
fprintf("Then, using the random row and column as the top left corner, a 36x36 sample was taken from the random image\n");
fprintf("This image was then stored in the folder 'cropped_training_images_notfaces', This whole process was repeated until there were the same amount of nonface images as face images\n\n");

fprintf("Once the folder was populated labels were applied to every image (face or nonface), the nonfaces and faces image sets were combined, randomized, and split into two new sets: training 80 percent of total and validation 20 of total\n\n");

fprintf("Features were then extracted from each image using the vl_hog function. To improve performance a 3x3 cellsize, as well as 27 orientations per feature were used which allowed for a more dense feature set\n");
fprintf("An SVM model was then trained on the training set, and then the validation set, both with a lambda value of 0.1\n\n");

fprintf("Results:\n");
