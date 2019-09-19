clear;

cub_path = '../../../raw_data/CUB_200_2011/CUB_200_2011/CUB_200_2011';
save_path = [cub_path,'/images'];

[idx,names] = textread([cub_path,'/images.txt'],'%s %s');
[idx,is_train] = textread([cub_path,'/train_test_split.txt'],'%s %s');

fid1 = fopen('train_imagelist.txt','w');
fid2 = fopen('val_imagelist.txt','w');

for i = 1 : length(idx)
    name = names{i};
    if(str2num(is_train{i}))
        fprintf(fid1,'%s %d\n',['/',names{i}],str2num(name(1:3)));
    else
        fprintf(fid2,'%s %d\n',['/',names{i}],str2num(name(1:3)));
    end
end

fclose(fid1);
fclose(fid2);