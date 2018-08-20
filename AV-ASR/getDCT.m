% function to extract discrete cosine transform from ultrasound images
% images are cropped by region of interest, and resized to 64x64, DCT applied, and added to a matrix

function s(sp)
spkrs=load('spkrs.mat'); % Matlab matrix with list of speakers
spkrs2=cell2mat(struct2cell(spkrs));
spk=spkrs2(sp).name;

for spkr={spk}
output_dir = 'train/';
fid = fopen([output_dir char(spkr) '-IDS-UXTD.txt'],'w'); % image IDs
image_dir = ['US_IMAGES/UXTD_TRAIN/' char(spkr)];

image_ext = '.png';
all_im_us3 = [];
 ults_matrix=[];
namelist=[];
listing=dir(image_dir);
listing=listing(~ismember({listing.name},{'.','..'}));

fprintf(1,'%i images found \n', length(listing));
  % Write data in output file
  save_ults_path=[output_dir, char(spkr),'_ET'];
  tic   
  for ind = 1:length(listing)
  im_us_gray3 = imread([image_dir '/' listing(ind).name]);
 % ROI
 im_us_gray_cropped3 = imcrop(im_us_gray3,[20 0 191 171]);
 % Image downsampling 
im_us_gray_cropped_resized3 = imresize(im_us_gray_cropped3,[64 64]);
% DCT2
J = dct2(im_us_gray_cropped_resized3);
dct_d=J(1:10,1:10);
dct_bits=dct_d(:);
% add to matrix
 ults_matrix=[ults_matrix,dct_bits];
 fprintf(fid,'%s\t',listing(ind).name);
  end
save(save_ults_path,'ults_matrix');
toc
   end
