function data = customreader(filename)
% filename
I = imread(filename);

[a,b,c] = size(I);

if c==1,
%     I0 = imread('0.jpg');
%     [indimg, indmap] = rgb2ind(I0,128);
%     indasgray = rgb2gray(reshape(indmap,[size(indmap,1),1,3]));
% %     image(indimg);
% %     colromap(indasgray);   %image painted gray but multiple ind might have same gray
%     I = round(ind2rgb(indimg, indmap)*254);   %not indasgray

I = grs2rgb(I);
end

data=imresize(I,[227 227]);
% whos data
