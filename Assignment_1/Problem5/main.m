clear;clc;
%% 1. set up vlfeat
run('./vlfeat-0.9.21/toolbox/vl_setup')

%% 2.read in the images
% you can change img here
img_left = imread("./img/left.jpg");
img_right = imread("./img/right.jpg");
% make single
img_left = im2single(img_left) ;
img_right = im2single(img_right) ;
% make grayscale
if size(img_left,3) > 1, img_left_g = rgb2gray(img_left) ; else img_left_g = img_left ; end
if size(img_right,3) > 1, img_right_g = rgb2gray(img_right) ; else img_right_g = img_right ; end

%% 3.use vl_sift to detect interest points and get sift descriptors
[F_left, D_left] = vl_sift(img_left_g);
[F_right, D_right] = vl_sift(img_right_g);

%% 4.match sift descriptors
[matches, scores] = vl_ubcmatch(D_left,D_right);
[~, index] = sort(scores, "descend");
matches = matches(:,index);
numMatches = size(matches,2);
match_left = F_left(1:2, matches(1,:)); 
match_right = F_right(1:2,matches(2,:));
X1 = match_left; X1(3,:) = 1;
X2 = match_right; X2(3,:) = 1;

%% 5.use RANSAC to choose descriptor-pairs and estimate homograpyh matrix
clear H score inlier ;
N = 150;
for t = 1:N
  % estimate homograpyh
  subset = vl_colsubset(1:numMatches, 4) ;
  A = [] ;
  for i = subset
    A = cat(1, A, kron(X1(:,i)', vl_hat(X2(:,i)))) ;
  end
  [U,S,V] = svd(A) ;
  H{t} = reshape(V(:,end),3,3);
  % score homography
  X2_ = H{t} * X1 ;
  du = X2_(1,:)./X2_(3,:) - X2(1,:)./X2(3,:) ;
  dv = X2_(2,:)./X2_(3,:) - X2(2,:)./X2(3,:) ;
  inlier{t} = (du.*du + dv.*dv) < 6*6 ;
  score(t) = sum(inlier{t}) ;
end

[score, best] = max(score) ;
H = H{best} ;
inlier = inlier{best} ;

%% 6. plot correspondences matches
dh1 = max(size(img_right,1)-size(img_left,1),0) ;
dh2 = max(size(img_left,1)-size(img_right,1),0) ;

figure;
imagesc([padarray(img_left,dh1,'post') padarray(img_right,dh2,'post')]) ;
o = size(img_left,2) ;
line([F_left(1,matches(1,inlier));F_right(1,matches(2,inlier))+o], ...
     [F_left(2,matches(1,inlier));F_right(2,matches(2,inlier))],...
     'Marker','o');
title(sprintf('%d (%.2f%%) inliner matches out of %d', ...
              sum(inlier), ...
              100*sum(inlier)/numMatches, ...
              numMatches)) ;

%% 7. stitch two images using H
box2 = [1  size(img_right,2)  size(img_right,2)   1 ;
        1  1                  size(img_right,1)   size(img_right,1) ;
        1  1                  1                   1                  ] ;
box2_ = inv(H) * box2 ;
box2_(1,:) = box2_(1,:) ./ box2_(3,:) ;
box2_(2,:) = box2_(2,:) ./ box2_(3,:) ;
ur = min([1 box2_(1,:)]):max([size(img_left,2) box2_(1,:)]) ;
vr = min([1 box2_(2,:)]):max([size(img_left,1) box2_(2,:)]) ;

[u,v] = meshgrid(ur,vr) ;
img_left_ = vl_imwbackward(im2double(img_left),u,v) ;

z_ = H(3,1) * u + H(3,2) * v + H(3,3) ;
u_ = (H(1,1) * u + H(1,2) * v + H(1,3)) ./ z_ ;
v_ = (H(2,1) * u + H(2,2) * v + H(2,3)) ./ z_ ;
img_right_ = vl_imwbackward(im2double(img_right),u_,v_) ;

mass = ~isnan(img_left_) + ~isnan(img_right_) ;
img_left_(isnan(img_left_)) = 0 ;
img_right_(isnan(img_right_)) = 0 ;
img_out = (img_left_ + img_right_) ./ mass ;

figure(2) ; clf ;
imagesc(img_out);
title('stitched image') ;