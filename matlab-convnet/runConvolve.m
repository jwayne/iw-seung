function target = runConvolve(kConv, data, kernel, task)
% (data,kernel) == (images, filters)
% (data,kernel) == (images, hidacts)
% (data,kernel) == (hidacts, filters)

if strcmp(task,'forward')
    numColors = size(data,5);
    if numColors == 1
        numFilters = size(kernel,1); filter_batch_size = 16;
        filter_batch_num = numFilters / filter_batch_size;
        assert(filter_batch_num == floor(filter_batch_num));
        
        numImages = size(data,1); imgSizeX = size(data,2); imgSizeY = size(data,3); imgSizeZ = size(data,4);
        filterSize =  size(kernel,2); 
        
        paddingStart = 0; moduleStride = 1; imgStride = numImages;
        numModulesX = imgSizeX - filterSize + 1; numModulesY = imgSizeY - filterSize + 1; numModulesZ = imgSizeZ - filterSize + 1;
        target = zeros(numImages, numModulesX, numModulesY, numModulesZ, numFilters, 'single');
        for b = 1 : filter_batch_num
            filterPerThread = 4; imagePerThread = 1;
            kConv.ThreadBlockSize = [32, 4];
            kConv.GridSize = [ceil(numImages/(32 * imagePerThread)), numModulesX * numModulesY * numModulesZ * filter_batch_size / (filterPerThread * 4) ]; % filterPerThread, BlockSize.Y
            target_gpu = feval(kConv,...
               target(:,:,:,:,(b-1)*filter_batch_size+1 : b*filter_batch_size), data, kernel((b-1)*filter_batch_size+1 : b*filter_batch_size,:,:,:,:),...
               numImages, filter_batch_size, imgSizeZ, imgSizeY, imgSizeX, filterSize, ...
               paddingStart, moduleStride, numModulesZ, numModulesY, numModulesX, imgStride);
           target(:,:,:,:,(b-1)*filter_batch_size+1 : b*filter_batch_size) = gather(target_gpu);
        end
    else
        numImages = size(data,1); imgSizeX = size(data,2); imgSizeY = size(data,3); imgSizeZ = size(data,4); 
        numFilters = size(kernel,1); filterSize =  size(kernel,2); 
        
        paddingStart = 0; moduleStride = 1; imgStride = numImages; numGroups = 1;
        numModulesX = imgSizeX - filterSize + 1; numModulesY = imgSizeY - filterSize + 1; numModulesZ = imgSizeZ - filterSize + 1;

        filterPerThread = 8; imagePerThread = 1;

        kConv.ThreadBlockSize = [32, 4];
        kConv.GridSize = [ceil(numImages/(32 * imagePerThread)), numModulesX * numModulesY * numModulesZ * numFilters / (filterPerThread * 4) ]; % filterPerThread, BlockSize.Y

        target = zeros(numImages, numModulesX, numModulesY, numModulesZ, numFilters, 'single');

        target_gpu = feval(kConv,...
           target, data, kernel,...
           numImages, numFilters, imgSizeZ, imgSizeY, imgSizeX, filterSize, ...
           paddingStart, moduleStride, numModulesZ, numModulesY, numModulesX, imgStride, numColors, numGroups);

        target = gather(target_gpu);
    end
    
elseif strcmp(task,'weight')
    numColors = size(data,5);
    if numColors == 1
        numImages = size(data,1); imgSizeX = size(data,2); imgSizeY = size(data,3); imgSizeZ = size(data,4); 
        numModulesX = size(kernel,2); numModulesY = size(kernel,3); numModulesZ = size(kernel,4); numFilters = size(kernel,5); 
        filterSize = imgSizeX - numModulesX + 1;
        paddingStart = 0; moduleStride = 1; imgStride = numImages; partialSum = numModulesX * numModulesY * numModulesZ;

        pixelsPerThread = 5; preLoadCases = 32; scaleOutput = 1 ./ (numImages * partialSum);

        kConv.ThreadBlockSize = [16, 8];
        kConv.GridSize = [numFilters*numModulesX*numModulesY*numModulesZ/partialSum/16, ceil(filterSize^3 /(8*pixelsPerThread))];

        target = zeros(numFilters, filterSize, filterSize, filterSize, numColors, 'single');
        target_gpu = feval(kConv, ....
           target, data, kernel,...
           numImages, numFilters, numModulesZ, numModulesY, numModulesX, imgSizeZ, imgSizeY, imgSizeX, ...
           filterSize, paddingStart, moduleStride, imgStride, partialSum, scaleOutput);

        target = gather(target_gpu);
    else
        numImages = size(data,1); imgSizeX = size(data,2); imgSizeY = size(data,3); imgSizeZ = size(data,4); 
        numFilters = size(kernel,5); numModulesX = size(kernel,2); numModulesY = size(kernel,3); numModulesZ = size(kernel,4); 
        filterSize = imgSizeX - numModulesX + 1;
        paddingStart = 0; moduleStride = 1; imgStride = numImages; partialSum = numModulesX * numModulesY * numModulesZ;

        preLoadCases = 32; filtersPerThread = 2; colorsPerThread = 8;
        scaleOutput = 1 ./ (numImages * partialSum); numGroups = 1;

        kConv.ThreadBlockSize = [16, 8];
        kConv.GridSize = [numFilters*numModulesX*numModulesY*numModulesZ/partialSum/16/filtersPerThread, ceil(filterSize^3 / 8) * (numColors / colorsPerThread)];

        target = zeros(numFilters, filterSize, filterSize, filterSize, numColors, 'single');
        target_gpu = feval(kConv, ....
           target, data, kernel,...
           numImages, numFilters, numModulesZ, numModulesY, numModulesX, imgSizeZ, imgSizeY, imgSizeX, ...
           filterSize, paddingStart, moduleStride, imgStride, numColors, numGroups, partialSum, scaleOutput);

        target = gather(target_gpu);
    end
elseif strcmp(task,'backward')
    numColors = size(kernel,5);
    if numColors == 1
        numImages = size(data,1); numModulesX = size(data,2); numModulesY = size(data,3); numModulesZ = size(data,4); moduleStride = 1;
        numFilters = size(kernel,1); filterSize = size(kernel,2); 

        imgSizeX = numModulesX + filterSize - 1; imgSizeY = numModulesY + filterSize - 1; imgSizeZ = numModulesZ + filterSize - 1; 
        paddingStart = 0; imgsPerThread = 2;

        kConv.ThreadBlockSize = [16, 16];
        kConv.GridSize = [ceil(numImages/(imgsPerThread *16)), imgSizeZ * ceil(imgSizeY/4) * ceil(imgSizeX/4)];

        target = zeros(numImages, imgSizeX, imgSizeY, imgSizeZ, numColors, 'single');
        target_gpu = feval(kConv,....
            target, data, kernel,...
            numModulesZ, numModulesY, numModulesX, numImages, numFilters, filterSize, ...
            imgSizeZ, imgSizeY, imgSizeX, paddingStart, moduleStride);

        target = gather(target_gpu);
    else
        numImages = size(data,1); numModulesX = size(data,2); numModulesY = size(data,3); numModulesZ = size(data,4); moduleStride = 1;
        numFilters = size(kernel,1); filterSize = size(kernel,2);

        imgSizeX = numModulesX + filterSize - 1; imgSizeY = numModulesY + filterSize - 1; imgSizeZ = numModulesZ + filterSize - 1; 
        paddingStart = 0; numGroups = 1;
        
        colorsPerThread = 4; imgsPerThread = 1;
        
        kConv.ThreadBlockSize = [32, 4];
        kConv.GridSize = [ceil(numImages/(imgsPerThread * 32)) * (numColors / (4 * colorsPerThread)), imgSizeZ * imgSizeY * imgSizeX];

        target = zeros(numImages, imgSizeX, imgSizeY, imgSizeZ, numColors, 'single');
        target_gpu = feval(kConv,....
            target, data, kernel,...
            numModulesZ, numModulesY, numModulesX, numImages, numFilters, filterSize, ...
            imgSizeZ, imgSizeY, imgSizeX, paddingStart, moduleStride, numColors, numGroups);

        target = gather(target_gpu);
    end
end
