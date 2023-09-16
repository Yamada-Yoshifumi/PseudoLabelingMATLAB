dataSetDir = 'train';
imageDir = fullfile(dataSetDir,'convolutions');
labelDir = fullfile(dataSetDir,'masks');

images = dir(fullfile(dataSetDir, "convolutions", "*.tif"));
labels = dir(fullfile(dataSetDir, "masks", "*.tif"));

max_x_translation = 20;
max_y_translation = 10;

for i = 1:length(images)
    disp(images(i).name);
    disp(labels(i).name);
    original_image = uint16(imread(fullfile(dataSetDir, "convolutions", images(i).name)));
    original_label = uint16(imread(fullfile(dataSetDir, "masks", labels(i).name)));
    x_translation = randi([-max_x_translation, max_x_translation]);
    y_translation = randi([-max_y_translation, max_y_translation]);
    cropped_image = [];
    cropped_label = [];
    if (x_translation > 0)
        if (y_translation > 0)
            cropped_image = original_image(x_translation:end, y_translation:end);
            cropped_label = original_label(x_translation:end, y_translation:end);
        else
            cropped_image = original_image(x_translation:end, 1:end+y_translation);
            cropped_label = original_label(x_translation:end, 1:end+y_translation);
        end
    else
        if (y_translation > 0)
            cropped_image = original_image(1:end+x_translation, y_translation:end);
            cropped_label = original_label(1:end+x_translation, y_translation:end);
        else
            cropped_image = original_image(1:end+x_translation, 1:end+y_translation);
            cropped_label = original_label(1:end+x_translation, 1:end+y_translation);
        end
    end
    cropped_label = imresize(cropped_label, size(original_label), "nearest");
    cropped_image = imresize(cropped_image, size(original_image), "nearest");
    imwrite(cropped_image, fullfile(dataSetDir, "convolutions", images(i).name));
    imwrite(cropped_label, fullfile(dataSetDir, "masks", labels(i).name));
end

imds = imageDatastore(imageDir);
imds.ReadFcn = @customReadDatastoreImage;
classNames = ["background", "cell"];
labelIDs = [0, 1];

%pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
pxds = imageDatastore(labelDir);
pxds.ReadFcn = @customReadPixelLabelDatastoreImage;

vImageDir = fullfile(dataSetDir,'v_convolutions');
vLabelDir = fullfile(dataSetDir,'v_masks');

vImds = imageDatastore(vImageDir);
vImds.ReadFcn = @customReadDatastoreImage;
classNames = ["background", "cell"];
labelIDs = [0, 1];

vPxds = pixelLabelDatastore(vLabelDir,classNames,labelIDs);
vPxds.ReadFcn = @customReadPixelLabelDatastoreImage;

imageSize = [256, 64];
numClasses = 2;
encoderDepth = 4;

lgraph = unetLayers(imageSize,numClasses,'EncoderDepth',encoderDepth);
weightedSegmentationLayer = WeightedSegmentationLayer("WeightedSegmentationLayer", 40);


% Initialize a variable to hold the sum of all images
sumImage = 0;

% Compute the sum of all images
for i = 1:length(imds.Files)
    img = readimage(imds, i);
    sumImage = sumImage + double(img);
end

% Compute the mean image
meanImage = sumImage / length(imds.Files);

inputLayer = imageInputLayer([256, 64, 1], 'Normalization', 'zerocenter', 'Name', 'ImageInputLayer', 'Mean', meanImage);
lgraph = replaceLayer(lgraph,"ImageInputLayer", inputLayer);
lgraph = removeLayers(lgraph, 'Segmentation-Layer');
%lgraph = removeLayers(lgraph, 'ImageInputLayer');
%lgraph = addLayers(lgraph, inputLayer);
%lgraph = connectLayers(lgraph, 'ImageInputLayer', 'Encoder-Stage-1-Conv-1');
%lgraph = addLayers(lgraph, weightedSegmentationLayer);
%lgraph = connectLayers(lgraph, 'Softmax-Layer', 'WeightedSegmentationLayer');

ds = combine(imds,pxds);
img=imds.read();
pimg = pxds.read();

vDs = combine(vImds, vPxds);

net = dlnetwork(lgraph);


imds = imageDatastore(imageDir);
imds.ReadFcn = @customReadDatastoreImage;
classNames = ["background", "cell"];
labelIDs = [0, 1];

%pxds = pixelLabelDatastore(labelDir,classNames,labelIDs);
pxds = imageDatastore(labelDir);
pxds.ReadFcn = @customReadPixelLabelDatastoreImage;

%s = load("DS/test_net_1.mat");
%net = s.test_net_1;
disp(net.Learnables);
numEpochs = 20;
miniBatchSize = 40;
mbq = minibatchqueue(ds,...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFormat={'SSBC', ''}, ...
    OutputEnvironment="auto");

initialLearnRate = 0.0001;
learningRate = initialLearnRate;
gradientDecayFactor = 0.001;
squaredGradientDecayFactor = 0.5;
momentum = 0.9;
trailingAvgGenerator = [];
trailingAvgSqGenerator = [];
velocity = [];

numIterationsPerEpoch = ceil(numel(imds.Files) / miniBatchSize);
numIterations = numEpochs * numIterationsPerEpoch;

monitor = trainingProgressMonitor(Metrics="Loss",Info=["Epoch","LearnRate"],XLabel="Iteration");

epoch = 0;
iteration = 0;

% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
    
    epoch = epoch + 1;

    % Shuffle data.
    shuffle(mbq);
    
    % Loop over mini-batches.
    while hasdata(mbq) && ~monitor.Stop
        
        iteration = iteration + 1;
        
        % Read mini-batch of data.
        % Read mini-batch of data and labels
        [X, T] = next(mbq);
        
        if size(X, 4) == miniBatchSize
            %Y_pred = forward(net, X); % Use network's forward method
        
            % Compute loss
            %loss = forwardLoss(weightedSegmentationLayer, Y_pred, T); % Use custom layer's forwardLoss method
            
            % Backward pass
            %dLdY = backwardLoss(weightedSegmentationLayer, Y_pred, T); % Use custom layer's backward method
            [loss,gradients,state] = dlfeval(@modelLoss,net, weightedSegmentationLayer,X,T);
            disp("loss");
            disp(loss);
            %disp("gradients");
            %disp(gradients);
            %disp("dLdY");
            %disp(dLdY);
            %learnRate = initialLearnRate/(1 + decay*iteration);
            % Update network parameters using optimization algorithm
            [net,trailingAvgDiscriminatorScale1,trailingAvgSqDiscriminatorScale1] = adamupdate(net,gradients,trailingAvgGenerator,trailingAvgSqGenerator,iteration, ...
                learningRate,gradientDecayFactor,squaredGradientDecayFactor);

            recordMetrics(monitor,iteration,Loss=loss);
            updateInfo(monitor,Epoch=epoch,LearnRate=learningRate);
            monitor.Progress = 100 * iteration/numIterations;
            %[loss,gradients, state] = dlfeval(@modelLoss,net, weightedSegmentationLayer,X,T);
            %net.State = state;
            %disp(loss);
            % Determine learning rate for time-based decay learning rate schedule.
            %learnRate = initialLearnRate/(1 + decay*iteration);
            
            % Update the network parameters using the SGDM optimizer.
            %[net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);
            
            % Update the training progress monitor.
            %recordMetrics(monitor,iteration,Loss=loss);
            %updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
            %monitor.Progress = 100 * iteration/numIterations;
        end
        % Convert input data to dlarray format
        %X = dlarray(miniBatchData, 'SSCB');
        

        
        %Y_pred = forward(net, X); % Use network's forward method
        
        % Compute loss
        %loss = forwardLoss(weightedSegmentationLayer, Y_pred, T); % Use custom layer's forwardLoss method
        
        % Backward pass
        %dLdY = backward(weightedSegmentationLayer, Y_pred, T); % Use custom layer's backward method
        
        % Update network parameters using optimization algorithm
        %net = updateParameters(net, dLdY, learnRate); 
        % Evaluate the model gradients, state, and loss using dlfeval and the
        % modelLoss function and update the network state.
        %[loss,gradients,state] = dlfeval(@modelLoss,net,X,T);
        %net.State = state;
        
        % Determine learning rate for time-based decay learning rate schedule.
        %learnRate = initialLearnRate/(1 + decay*iteration);
        
        % Update the network parameters using the SGDM optimizer.
        %[net,velocity] = sgdmupdate(net,gradients,velocity,learnRate,momentum);
        
        % Update the training progress monitor.
        %recordMetrics(monitor,iteration,Loss=loss);
        %updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
        %monitor.Progress = 100 * iteration/numIterations;
    end
end

test_net_1 = net;
save("DS/test_net_1.mat", "test_net_1");

function [loss,gradients, state] = modelLoss(net, weightedSegmentationLayer, X, T)

    % Forward data through network.
    [Y, state] = forward(net, X);
    %Y = weightedSegmentationLayer.predict(Y_pre);
    %disp(T(100:160,20:35,1));
    %disp(Y(100:160,20:35,2,1))
    % Calculate cross-entropy loss.
    loss = weightedSegmentationLayer.forwardLoss(Y,T);
    %Y = squeeze(Y(:, :, 2, :));
    %disp(Y(100:159,20:35,2));
    %disp(T(100:159,20:35,1))
    %loss = crossentropy(Y, T);
    %loss = sum((Y - T).^2, "all");
    % Calculate gradients of loss with respect to learnable parameters.
    %gradients = weightedSegmentationLayer.backwardLoss(Y,T);
    
    gradients = dlgradient(loss,net.Learnables);
    %disp(gradients.Value);
end

function X = preprocessMiniBatchPredictors(dataX)

    % Concatenate.
    X = cat(4,dataX{1:end});

end

function [X,T] = preprocessMiniBatch(dataX,dataT)
    
    % Preprocess predictors.
    X = preprocessMiniBatchPredictors(dataX);
    
    % Extract label data from cell and concatenate.
    T = cat(2,dataT{1:end});
    
    % One-hot encode labels.
    T = onehotencode(T,1);

end

function Y = modelPredictions(net,mbq,classes)

    Y = [];
    
    % Loop over mini-batches.
    while hasdata(mbq)
        X = next(mbq);
    
        % Make prediction.
        scores = predict(net,X);
    
        % Decode labels and append to output.
        labels = onehotdecode(scores,classes,1)';
        Y = [Y; labels];
    end

end
%{
options = trainingOptions("adam", ...
    InitialLearnRate=1e-4, ...
    SquaredGradientDecayFactor=0.99, ...
    MaxEpochs=20, ...
    MiniBatchSize=40, ...
    Plots="training-progress", ...
    Shuffle="every-epoch", ...
    ValidationFrequency=50, ...
    ValidationData=vDs);

net = trainNetwork(ds,lgraph,options);
test_net_1 = net;
save("DS/test_net_1.mat", "test_net_1");
%}
function data = customReadDatastoreImage(filename)
    % code from default function: 
    onState = warning('off', 'backtrace');
    c = onCleanup(@() warning(onState)); 
    img = imread(filename); % added lines:
    img = imresize(img,[256 64]);
    img = im2double(img);
    pseudoLabeling = PseudoLabeling();
    data = pseudoLabeling.normalize99(img);
end

function data = customReadPixelLabelDatastoreImage(filename)
    % code from default function: 
    onState = warning('off', 'backtrace'); 
    c = onCleanup(@() warning(onState)); 
    img = imread(filename); % added lines:
    img(img > 0.5) = 1;
    data = imresize(img,[256 64]);
    pseudoLabeling = PseudoLabeling();
    data = pseudoLabeling.erode(data, [3 3]);
    data = uint8(data);
end