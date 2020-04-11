run('../vlfeat-0.9.21/toolbox/vl_setup')
load('svm.mat');
imageDir = 'test_images';
imageList = dir(sprintf('%s/*.jpg',imageDir));
nImages = length(imageList);

bboxes = zeros(0,4);
confidences = zeros(0,1);
image_names = cell(0,1);
  
cellSize = 3;
dim = 36;
for i=1:nImages
%     bboxes = zeros(0,4);
%     confidences = zeros(0,1);
%     image_names = cell(0,1);

    % load and show the image
    im = im2single(imread(sprintf('%s/%s',imageDir,imageList(i).name)));
    imshow(im);
    hold on;
    
    % generate a grid of features across the entire image. you may want to 
    % try generating features more densely (i.e., not in a grid)
    feats = vl_hog(im,cellSize,'numOrientations',21);
    
    % concatenate the features into 6x6 bins, and classify them (as if they
    % represent 36x36-pixel faces)
    [rows,cols,~] = size(feats);
    confs = zeros(rows,cols);
    
    bin = zeros(12,12,67);
    for r=1:rows-11
        for c=1:cols-11
            bin(:,:,:) = feats(r:r+11,c:c+11,:);
            
            % create feature vector for the current window and classify it using the SVM model,
            % take dot product between feature vector and w and add b,
            % store the result in the matrix of confidence scores confs(r,c)
            
            vect = reshape(bin,1,9648);
            confs(r,c) = vect*w+b;
        end
    end
       
    % get the most confident predictions 
    [~,inds] = sort(confs(:),'descend');
    inds = inds(1:20); % (use a bigger number for better recall)  
    for n=1:numel(inds)        
        [row,col] = ind2sub([size(feats,1) size(feats,2)],inds(n));
        
        bbox = [ col*cellSize ...
                 row*cellSize ...
                (col+12-1)*cellSize ...
                (row+12-1)*cellSize];
        conf = confs(row,col);
        image_name = {imageList(i).name};
        
%         % plot
%         plot_rectangle = [bbox(1), bbox(2); ...
%             bbox(1), bbox(4); ...
%             bbox(3), bbox(4); ...
%             bbox(3), bbox(2); ...
%             bbox(1), bbox(2)];
%         plot(plot_rectangle(:,1), plot_rectangle(:,2), 'r-');
        
        % save         
        bboxes = [bboxes; bbox];
        confidences = [confidences; conf];
        image_names = [image_names; image_name];
    end
    
    %Non Max suppression
    bboxes2 = zeros(0,4);   %Best bboxes
    lim = size(bboxes,1) + 1;   %iterate over all bboxes
    i2 = 1;
    while i2 < lim
        bb1 = bboxes(i2,:); 
        bboxes2 = [bboxes2; bb1];   %add bbox to bboxes2
        
        j = i2 + 1; %Second loop starting point
        while j < lim
            bb2 = bboxes(j,:);
            bi=[max(bb1(1),bb2(1)) ; max(bb1(2),bb2(2)) ; min(bb1(3),bb2(3)) ; min(bb1(4),bb2(4))];
            iw = bi(3) - bi(1) + 1;
            ih = bi(4) - bi(2) + 1;
            if iw>0 && ih>0 %Check for intersection
                bboxes(j,:) = [];   %Remove intersecting bbox with lower confidence
                confidences(j,:) = [];
                image_names(j,:) = [];
                j = j - 1;  %Reset j for updated bboxes
                lim = size(bboxes,1) + 1;   %Reset lim for updated bboxes
            end
            j = j + 1;  %Increase second loop iterator
        end
        lim = size(bboxes,1) + 1;   %Reset lim for updated bboxes
        i2 = i2 + 1;    %Increase first loop iterator
    end
    
%     %Print kept bboxes
%     for n = 1:size(bboxes2(:,1))
%         plot_rectangle = [bboxes2(n,1), bboxes2(n,2); ...
%         bboxes2(n,1), bboxes2(n,4); ...
%         bboxes2(n,3), bboxes2(n,4); ...
%         bboxes2(n,3), bboxes2(n,2); ...
%         bboxes2(n,1), bboxes2(n,2)];
%         plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
%     end
%     pause;
    fprintf('got preds for image %d/%d\n', i,nImages);
end


% % evaluate
label_path = 'test_images_gt.txt';
[gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections] = ...
    evaluate_detections_on_test(bboxes, confidences, image_names, label_path);


%Class.jpg
classim = im2single(imread('class.jpg'));
imshow(classim);
hold on;

feats = vl_hog(classim,cellSize,'numOrientations',21);

class_bboxes = zeros(0,4);
class_confidences = zeros(0,1);

[rows,cols,~] = size(feats);
confs = zeros(rows,cols);
    
 bin = zeros(12,12,67);
 for r=1:rows-11
    for c=1:cols-11
        bin(:,:,:) = feats(r:r+11,c:c+11,:);
        vect = reshape(bin,1,9648);
        confs(r,c) = vect*w+b;
    end
 end
 
 % get the most confident predictions
 [~,inds] = sort(confs(:),'descend');
 inds = inds(1:20); % (use a bigger number for better recall)  
 for n=1:numel(inds)        
     [row,col] = ind2sub([size(feats,1) size(feats,2)],inds(n));
        
     bbox = [ col*cellSize ...
              row*cellSize ...
              (col+12-1)*cellSize ...
              (row+12-1)*cellSize];
     conf = confs(row,col);
       
     % plot
     plot_rectangle = [bbox(1), bbox(2); ...
            bbox(1), bbox(4); ...
            bbox(3), bbox(4); ...
            bbox(3), bbox(2); ...
            bbox(1), bbox(2)];
     plot(plot_rectangle(:,1), plot_rectangle(:,2), 'r-');
        
     % save         
     class_bboxes = [class_bboxes; bbox];
     class_confidences = [class_confidences; conf];
 end
 
 %Non Max suppression
 bboxes2 = zeros(0,4);   %Best bboxes
 lim = size(class_bboxes,1) + 1;   %iterate over all bboxes
 i2 = 1;
 while i2 < lim
     bb1 = class_bboxes(i2,:); 
     bboxes2 = [bboxes2; bb1];   %add bbox to bboxes2
       
     j = i2 + 1; %Second loop starting point
     while j < lim
         bb2 = class_bboxes(j,:);
         bi=[max(bb1(1),bb2(1)) ; max(bb1(2),bb2(2)) ; min(bb1(3),bb2(3)) ; min(bb1(4),bb2(4))];
         iw = bi(3) - bi(1) + 1;
         ih = bi(4) - bi(2) + 1;
         if iw>0 && ih>0 %Check for intersection
             class_bboxes(j,:) = [];   %Remove intersecting bbox with lower confidence
             class_confidences(j,:) = [];
             j = j - 1;  %Reset j for updated bboxes
             lim = size(class_bboxes,1) + 1;   %Reset lim for updated bboxes
         end
         j = j + 1;  %Increase second loop iterator
     end
     lim = size(class_bboxes,1) + 1;   %Reset lim for updated bboxes
     i2 = i2 + 1;    %Increase first loop iterator
 end
    
 %Print kept bboxes
 for n = 1:size(bboxes2(:,1))
     plot_rectangle = [bboxes2(n,1), bboxes2(n,2); ...
     bboxes2(n,1), bboxes2(n,4); ...
     bboxes2(n,3), bboxes2(n,4); ...
     bboxes2(n,3), bboxes2(n,2); ...
     bboxes2(n,1), bboxes2(n,2)];
     plot(plot_rectangle(:,1), plot_rectangle(:,2), 'g-');
 end
 pause;
