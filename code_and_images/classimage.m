%Class.jpg
classim = im2single(imread('class.jpg'));
imshow(classim);
hold on;

feats = vl_hog(classim,3,'numOrientations',21);

bboxes = zeros(0,4);
confidences = zeros(0,1);

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
     bboxes = [bboxes; bbox];
     confidences = [confidences; conf];
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
             j = j - 1;  %Reset j for updated bboxes
             lim = size(bboxes,1) + 1;   %Reset lim for updated bboxes
         end
         j = j + 1;  %Increase second loop iterator
     end
     lim = size(bboxes,1) + 1;   %Reset lim for updated bboxes
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
