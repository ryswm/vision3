%Ryan Woodworth 500752821
%Akash Chhabria 500763706

run('../vlfeat-0.9.21/toolbox/vl_setup')
load('feats_labels.mat')

% feats = cat(1,pos_feats,neg_feats);
% labels = cat(1,ones(pos_nImages,1),-1*ones(neg_nImages,1));

lambda = 0.1;
[w,b] = vl_svmtrain(training_feats',training_labels,lambda);

fprintf('Classifier performance on train data:\n')
confidences = training_feats*w + b;

[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, training_labels);

[wv,bv] = vl_svmtrain(validation_feats',valid_labels,lambda);

fprintf('Classifier performance on validation data:\n')
confidences = validation_feats*wv + bv;

[tp_rate2, fp_rate2, tn_rate2, fn_rate2] =  report_accuracy(confidences, valid_labels);

save('svm.mat','w','b', 'wv','bv', 'b');
