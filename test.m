%{
filelist = dir("test/");
test_set = zeros(256, 64, size(filelist, 1)-2);

digitDatasetPath = fullfile("test/");
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true);

imds.ReadFcn = @customReadDatastoreImage;

classNames = ["cell" "background"];
labelIDs = [255 0];

net = load("DS/test_net.mat").test_net_1;
pxdsResults = semanticseg(imds,net,"WriteLocation", "test/", "NamePrefix", "", "NameSuffix", "", "OutputFolderName", "masks");

function data = customReadDatastoreImage(filename)
    % code from default function: 
    onState = warning('off', 'backtrace'); 
    c = onCleanup(@() warning(onState)); 
    data = imread(filename); % added lines:
    data = imresize(data,[256 64]);
    data = im2double(data);
    pseudoLabeling = PseudoLabeling();
    data = pseudoLabeling.normalize99(data);
end
%}

inputFolder = 'test';
outputFolder = 'test/masks';
mkdir(outputFolder); % Create the output folder if it doesn't exist
s = load("DS/test_net_1.mat");
net = s.test_net_1;
%classes = load("DS/test_net_1.mat").classes;

weightedSegLayer = WeightedSegmentationLayer("Output_Layer", 40);
imageFiles = fullfile(inputFolder); % List all JPG files in input folder
imageFileNames = dir( fullfile(inputFolder));
imds = imageDatastore(imageFiles);

miniBatchSize = 1;
imds.ReadSize = miniBatchSize;

mbq = minibatchqueue(imds,...
    "MiniBatchSize",miniBatchSize,...
    "MiniBatchFcn", @preprocessMiniBatch,...
    "MiniBatchFormat","SSBC");

i = 1;

while hasdata(mbq)

    % Read input image
    %inputImage = imread(fullfile(inputFolder, imageFiles(i).name));
    %data = imresize(inputImage,[256 64]);
    %data = im2double(data);
    %pseudoLabeling = PseudoLabeling();
    %data = dlarray(pseudoLabeling.normalize99(data));
    
    data = next(mbq);

    % Perform Semantic Segmentation
    %segmentationResult = semanticseg(data, net); % 'net' is your pretrained network
    YPred = predict(net, data);
    dlSegmentationResult = weightedSegLayer.predict(YPred);
    % Extract base file name without extension
    [~, baseFileName, ~] = fileparts(imageFileNames(i).name);
    
    % Save Segmentation Result as TIFF with original file name
    outputImageFilename = fullfile(outputFolder, [baseFileName '.tif']);
    segmentationResult = extractdata(dlSegmentationResult);
    segmentationResultCpu = gather(segmentationResult);
    segmentationResultCleaned = bwareaopen(uint8(segmentationResultCpu), 80);
    segmentationResultCleaned = imfill(uint8(segmentationResultCleaned));
    imwrite(uint8(segmentationResultCleaned), outputImageFilename);
    i = i+1;
end

function data = preprocessMiniBatch(cell)
    data = cell{:};
    data = imresize(data,[256 64]);
    pseudoLabeling = PseudoLabeling();
    data = pseudoLabeling.normalize99(data);
    data = cat(3, data);
end