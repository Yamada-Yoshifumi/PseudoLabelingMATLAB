classdef WeightedSegmentationLayer < nnet.layer.ClassificationLayer % ...
        % & nnet.layer.Acceleratable % (Optional)
        
    properties
        % (Optional) Layer properties.
        batch_size {mustBeInteger};
        % Layer properties go here.
    end
 
    methods
        function layer = WeightedSegmentationLayer(name, batch_size)           
            % (Optional) Create a myClassificationLayer.
            layer.Name = name;
            layer.batch_size = batch_size;
            % Layer constructor function goes here.
        end
        function distance_transform = custom_bwdist(layer, binary_image)
        
            % Initialize the output distance transform image with NaNs
            distance_transform = zeros(size(binary_image));
        
            % Get the dimensions of the input binary image
            [rows, cols] = size(binary_image);
        
            % Find the indices of non-zero pixels in the binary image
            [non_zero_rows, non_zero_cols] = find(binary_image);
        
            % Process each pixel and calculate the distance to the nearest non-zero pixel
            for r = 1:rows
                for c = 1:cols
                    % If the current pixel is non-zero, set its distance to zero
                    if binary_image(r, c)
                        distance_transform(r, c) = 0;
                    else
                        % Calculate the distance to the nearest non-zero pixel using Euclidean distance
                        distances = sqrt((r - non_zero_rows).^2 + (c - non_zero_cols).^2);
                        distance_transform(r, c) = min(distances);
                    end
                end
            end
        end
        function w = unet_weight_map(layer, y, wc, w0, sigma)
           y_cp = gather(y);
           y_int_cp = extractdata(y_cp);
           labels = bwlabel(y_int_cp, 4);
           I2 = cast(labels, "uint8");
           imwrite(I2,'mask.png');
           no_labels = labels == 0;
           label_ids = unique(labels);
           label_ids = sort(label_ids);
           label_ids = label_ids(2:end);
           if numel(label_ids) > 1
               distances = zeros(size(y, 1), size(y, 2), numel(label_ids));
               
               for i = 1:numel(label_ids)
                   label_id = label_ids(i);
                   selected_foreground = labels == label_id;
                   cpu_selected_foreground = gather(selected_foreground);
                   D = bwdist(cpu_selected_foreground, "quasi-euclidean");
                   %D = gpuArray(D);
                   %D = custom_bwdist(cpu_selected_foreground);
                   %save("transform.mat", "selected_foreground");
                   distances(:, :, i) = D;
                   %w_mapped = 255 * (D - prctile(D, 0.01, "all"))/(prctile(D, 99.99, "all") - prctile(D, 0.01, "all"));
                   %I2 = cast(D, "uint8");
                   %imwrite(I2,'myImageNoClassWeights.png');
               end
               
               distances = sort(distances, 3);
               d1 = distances(:, :, 1);
               d2 = distances(:, :, 2);
               w = w0 * exp(-1/2 * ((d1 + d2) / sigma).^2) .* no_labels;
           else
               w = zeros(size(y));
           end
           if ~isempty(wc)
               class_weights = zeros(size(y));
               
               for k = 1:numel(wc(:,1))
                   label = wc(k, 1);
                   weight = wc(k, 2);
                   class_weights(y == label) = weight;
               end
               
               w = w + class_weights;
           end
           w_mapped = 255 * (w - min(w, [], "all"))/(max(w, [], "all") - min(w, [], "all"));
           I2 = cast(w_mapped, "uint8");
           imwrite(I2,'myImage.png');
        end
        function Y = predict(layer, X)
            %Y = zeros(size(X(1, :, :)));
            Y = squeeze(X(:,:,1) < X(:,:,2));
        end
        function loss = forwardLoss(layer, Y, T)
            %T_results = squeeze(T(:, :, 2, :));
            %Y_results = squeeze(Y(:, :, 1, :) < Y(:, :, 2, :));
            Y_results = squeeze(Y(:,:,2,:));
            T_results = squeeze(T);
            %Y_results = squeeze(Y_results);

            % Make background weights be equal to the model's prediction
            weight = ones(size(T_results));
            
            for i = 1:layer.batch_size
                weight(:,:,i) = layer.unet_weight_map(T_results(:,:,i), [0 5 ; 1 5], 10, 5);
            end
            
            %bool_bkgd = weight == 0 / 255;
            %weight = bool_bkgd .* Y_results + ~bool_bkgd .* weight;
            
            epsilon = 1e-7;
            Y_results = max(min(Y_results, 1 - epsilon), epsilon);
            Y_results = log(Y_results ./ (1 - Y_results));
            zeros_copy = zeros(size(Y_results), 'like', Y_results);
            cond = Y_results >= zeros_copy;
            relu_logits = cond .* Y_results;
            
            neg_abs_logits = ~cond .* Y_results;
            entropy = relu_logits - Y_results .* T_results + log(exp(neg_abs_logits) + 1);
            loss = mean(weight .* entropy, 'all');
            loss = 1e6 * (1 / sqrt(sum(weight, "all"))) * loss;
            %{
            sumSquares = sum((Y-T).^2);
    
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = mean(sum(sumSquares)/N, "all");
            %}
        end
        function dLdY = backwardLoss(layer, Y, T)
            T_results = squeeze(T);
            Y_results = squeeze(Y(:, :, 2, :));

            
            % Make background weights be equal to the model's prediction
            weight = ones(size(T_results));
            
            for i = 1:layer.batch_size
                weight(:,:,i) = layer.unet_weight_map(T_results(:,:,i), [0 5 ; 1 5], 10, 5);
                %weight(:,:, 2,i) = layer.unet_weight_map(T_results(:,:,i), [0 1 ; 1 5], 10, 5);
            end
            dLdY = squeeze(sum(2*(Y_results-T_results).*weight, 3))/layer.batch_size;
        end
    end
end