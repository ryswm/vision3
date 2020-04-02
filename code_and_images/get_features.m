close all
clear
run('../vlfeat-0.9.21/toolbox/vl_setup')

load('sets.mat');

% pos_imageDir = 'cropped_training_images_faces';
% pos_imageList = dir(sprintf('%s/*.jpg',pos_imageDir));
% pos_nImages = length(pos_imageList);
% 
% neg_imageDir = 'cropped_training_images_notfaces';
% neg_imageList = dir(sprintf('%s/*.jpg',neg_imageDir));
% neg_nImages = length(neg_imageList);

cellSize = 3;
featSize = 31*144;%cellSize^2;

trainingL = length(training);

training_feats = zeros(trainingL,featSize);
for i=1:trainingL
    im = im2single(imread(sprintf('%s/%s',training(i).folder,training(i).name)));
    feat = vl_hog(im,cellSize);
    %disp(size(feat));
    training_feats(i,:) = feat(:);
    fprintf('got feat for training image %d/%d\n',i,trainingL);
     imhog = vl_hog('render', feat);
     subplot(1,2,1);
     imshow(im);
     subplot(1,2,2);
     imshow(imhog)
     pause;
end

%  validation_feats = zeros(neg_nImages,featSize);
% for i=1:neg_nImages
%     im = im2single(imread(sprintf('%s/%s',neg_imageDir,neg_imageList(i).name)));
%     feat = vl_hog(im,cellSize);
%     neg_feats(i,:) = feat(:);
%     fprintf('got feat for neg image %d/%d\n',i,neg_nImages);
% %     imhog = vl_hog('render', feat);
% %     subplot(1,2,1);
% %     imshow(im);
% %     subplot(1,2,2);
% %     imshow(imhog)
% %     pause;
% end
