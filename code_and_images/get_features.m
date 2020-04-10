close all
clear
run('../vlfeat-0.9.21/toolbox/vl_setup')

load('sets.mat');


cellSize = 3;%6;
featSize = 31*144;%cellSize^2;

trainingL = length(training);
validationL = length(validation);

training_feats = zeros(trainingL,featSize);
for i=1:trainingL
    im = im2single(imread(sprintf('%s/%s',training(i).folder,training(i).name)));
    feat = vl_hog(im,cellSize);
    training_feats(i,:) = feat(:);
    fprintf('got feat for training image %d/%d\n',i,trainingL);
%      imhog = vl_hog('render', feat);
%      subplot(1,2,1);
%      imshow(im);
%      subplot(1,2,2);
%      imshow(imhog)
%      pause;
end

validation_feats = zeros(validationL,featSize);
for i=1:validationL
    im = im2single(imread(sprintf('%s/%s',validation(i).folder,validation(i).name)));
    feat = vl_hog(im,cellSize);
    validation_feats(i,:) = feat(:);
    fprintf('got feat for validation image %d/%d\n',i,validationL);
%     imhog = vl_hog('render', feat);
%     subplot(1,2,1);
%     imshow(im);
%     subplot(1,2,2);
%     imshow(imhog)
%     pause;
end

save('feats_labels.mat','training_feats','validation_feats','training_labels','valid_labels');
