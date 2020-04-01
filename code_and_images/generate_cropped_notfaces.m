% you might want to have as many negative examples as positive examples
n_have = 0;
n_want = numel(dir('cropped_training_images_faces/*.jpg'));

imageDir = 'images_notfaces';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

new_imageDir = 'cropped_training_images_notfaces';
mkdir(new_imageDir);

dim = 36;

while n_have < n_want
    % generate random 36x36 crops from the non-face images
    
    %Get random image from not_faces
    ind = randi([1 nImages], 1, 1);
    tempImg = imread(append(imageList(ind).folder, '/', imageList(ind).name));
    tempImg = im2double(rgb2gray(tempImg));
    
    %Make temp 36x36 image
    tempCrop = zeros(dim,dim);
    
    %Get 36x36 cropped image from tempImg
    [row, col, ~] = size(tempImg);
    randOrigRow = randi([1 (row - dim)], 1,1);
    randOrigCol = randi([1 (col - dim)], 1,1);
    
    tempCrop(1:36,1:36) = tempImg(randOrigRow:randOrigRow + 35, randOrigCol:randOrigCol + 35);
    path = append(new_imageDir,'/',num2str(n_have),'.jpg');
    imwrite(tempCrop,path);
    
    %Increase loop
    n_have = n_have + 1;
end

%Split all cropped images into training and validation sets


