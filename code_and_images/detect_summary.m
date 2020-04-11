fprintf("Part 2: Our Approach\n");
fprintf("For Part 2 we used a very similar approach to part 1. In order to use our trained model, a 3x3 cell and 21 orientations per feature were used.\n");
fprintf("In order for our window to cover 36x36 pixels 12x12 bins were taken for each sliding window position (due to our smaller feature cell).\n");
fprintf("This resulted in feature vector the size of of 1x9648 per window per image\n");
fprintf("Using this feature vector for each window position and the SVM model, a prediction was calculated along with its confidence value.\n");
fprintf("(The closer to 1, the more likely a face is within this window position)\n\n");

fprintf("Once the confidence was calculated for each window position.\n");
fprintf("The values were sorted in descending order and the top 20 confidences with their associated bounding boxes were taken.\n");
fprintf("We accounted for overlapping predictions by running the 20 taken window positions through a non-maximum suppression algorithm.\n\n");

fprintf("\n");

fprintf("The result for the average precision is: %d.\n\n", 0.068);

avg_precs = imread('average_precision.png');
imshow(avg_precs);

fprintf("Our class.jpg predictions did not do as well as we would have hoped, but it id entirely due to our lack of predictions at different scales\n");
fprintf("However, we did recieve similar accuracy to our testing results, with only 3 faces identified.");