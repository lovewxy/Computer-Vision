clear;clc;
%% 1. read in the image
% you can change img here
img = imread("./img/img1.jpg");
% img = imread("./img/img2.jpg");
% make single
img = im2single(img) ;
% make grayscale
if size(img,3) > 1, img = rgb2gray(img) ; else img = img ; end

%% 2.set up the params
sigma0 = 1;
numScales = 6;
scale_ratio = 2;
thresh = 0.005;

%% 3.calculate scale space
[M, N] = size(img);
scale_space = zeros(M, N, numScales);
for i = 1:numScales
    sigma = sigma0 * scale_ratio^(i-1);
    filter_size = 3 * sigma + 1;    % change pattern size according to sigma
    LoG_filter = (sigma ^ 2) * fspecial('log', filter_size, sigma); 
    img_log_filtered = imfilter(img, LoG_filter, 'replicate');   
    scale_space(:, :, i) = img_log_filtered;
end

%% 4.calculate local maximum in space scale 
scale_space = scale_space .^ 2;     % increase difference
max_values = zeros(M, N, numScales);
for i = 1:numScales
    max_values(:, :, i) = ordfilt2(scale_space(:, :, i), numScales^2, ones(numScales,numScales));
end

%% 5.get local maximum in sigma scale
blobs = zeros(M, N, numScales);
for i = 1:M
    for j = 1:N
        max_value = max(max_values(i, j, :));
        if max_value < thresh
            blobs(i, j, :) = 0;
        else
            blobs(i, j, :) = max_value;
        end
    end
end
blobs = blobs .* (blobs == scale_space);

%% 6.record information from (x,y,sigma) space
rows = [];
cols = [];
radiuses = [];
for i=1:numScales
    [row, col] = find(blobs(:, :, i));
    rows = [rows; row];
    cols = [cols; col];
    temp_redius = sigma0 * scale_ratio.^(i-1) * sqrt(2);    % usually, the ratio between adjacent is set to sqrt(2)
    radius = repmat(temp_redius, [size(row,1),1]);          % copy rows times
    radiuses = [radiuses; radius];
end

%% 7.Visualize the feature points
figure;
imshow(img);
hold on;

theta = 0:pi/30:2*pi;
X = rows + sin(theta) .* radiuses;
Y = cols + cos(theta) .* radiuses;
line(Y', X', 'Color', 'r');

title(sprintf('%d feature points', size(radiuses, 1)));