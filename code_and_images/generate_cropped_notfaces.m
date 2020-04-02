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

%Split all cropped images into training and validation set
imageDir2 = 'cropped_training_images_faces';
all_cropped = cat(1, dir(sprintf('%s/*.jpg','cropped_training_images_notfaces')), dir(sprintf('%s/*.jpg',imageDir2)));

[rows, ~] = size(all_cropped);
perm = randperm(rows);
training = all_cropped(perm(1:round(rows*0.8)),:);
validation = all_cropped(perm((round(rows*0.8) + 1):end),:);


%Assign labels
[trainrows, traincols] = size(training);
training_labels = zeros(trainrows,1);
for i = 1:trainrows
   if strcmp(training(i).folder, append(pwd,'/','cropped_training_images_faces'))
       training_labels(i) = 1;
   elseif strcmp(training(i).folder, append(pwd,'/','cropped_training_images_notfaces'))
       training_labels(i) = -1;
   end
end

[validrows, validcols] = size(validation);
valid_labels = zeros(validrows,1);
for i = 1:validrows
   if strcmp(validation(i).folder, append(pwd,'/','cropped_training_images_faces'))
       valid_labels(i) = 1;
   elseif strcmp(validation(i).folder, append(pwd,'/','cropped_training_images_notfaces'))
       valid_labels(i) = -1;
   end
end

save('sets.mat','training','validation','training_labels','valid_labels');

