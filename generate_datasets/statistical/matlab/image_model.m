classdef image_model
    % A progression of natural image models:
    % Power spectrum
    % http://web.mit.edu/torralba/www/ne3302.pdf
    % https://www.cs.unm.edu/~williams/cs591/ne940406.pdf
   methods (Static)
       function c = get_common_args(resolution, experiment_dataset_dir, ...
            impose_spectrum, impose_wmm, impose_color_model)

            c.imagesize = resolution; % Size of generated images (square images)
            c.Nor=4;      % Size of the steerable pyramid...
            if c.imagesize == 128
                c.Nscales=3;  %    ...(you can only do 3 scales if the image is 128x128)
            elseif c.imagesize == 256
                c.Nscales=4;  %    ...(you can only do 3 scales if the image is 128x128)
            end
            c.sloperange = [0.5 3.5]; % range of slope values for the power spectrum
            c.Niter = 10;   % Number of iterations used to set the image model parameters

            dataset_dir = strcat(experiment_dataset_dir, '_', sprintf('%03d', c.imagesize));
            if impose_spectrum
                dataset_dir = strcat(dataset_dir, '_impose_spectrum');
            end
            if impose_color_model
                dataset_dir = strcat(dataset_dir, '_impose_color_model');
            end
            if impose_wmm
                dataset_dir = strcat(dataset_dir, '_impose_wmm');
            end
            c.dataset_dir = strcat(dataset_dir, '/train');

            c.impose_spectrum = impose_spectrum;
            c.impose_wmm = impose_wmm;
            c.impose_color_model = impose_color_model;
       end

    %% Experiment 1: generate synthetic images by using a statistical image model with parameters seleted by hand
    function generate_projection_dataset(sample_list, resolution, base_dataset_dir, ...
            impose_spectrum, impose_wmm, impose_color_model, correlate_channels, jpg)
        experiment_dataset_dir = strcat(base_dataset_dir, '/low_dimensional_parametric_image_model');
        if correlate_channels
            experiment_dataset_dir = strcat(experiment_dataset_dir, '_correlate_channels');
        end
        if jpg
            experiment_dataset_dir = strcat(experiment_dataset_dir, '_jpg');
        end
        p = image_model.get_common_args(resolution, experiment_dataset_dir, ...
            impose_spectrum, impose_wmm, impose_color_model);
        p.correlate_channels = correlate_channels;
        parfor i = 1:size(sample_list,2)
            sample_i = sample_list(i);
            subdir = sprintf('%08d', floor(sample_i / 1000) * 1000);
            folder = strcat(p.dataset_dir, '/', subdir);
            if jpg
                image_file = strcat(folder, sprintf('/img_%06d.jpg', sample_i));
            else
                image_file = strcat(folder, sprintf('/img_%06d.png', sample_i));
            end
            if ~isfile(image_file)
                i
                image_file
                try
                    if ~exist(folder, 'dir')
                       mkdir(folder);
                    end
                    image = image_model.generate_single_image_model_projection_dataset(sample_i, p);
                    if ~isfile(image_file)
                       imwrite(image, image_file);
                    end
                catch exception
                   exception
                   nothing = 0;
                end
            end
        end
    end
    function image = generate_single_image_model_projection_dataset(sample_i, p)
        rand('seed', sample_i); randn('seed',sample_i);
        %Sample synthetic image using an statistical image model
        if p.impose_color_model
            [p.colorPC, p.MeanColor, p.HistIma, p.HistImaBins, img] = sample_colormodel(p.imagesize);
        elseif p.correlate_channels
            [p.colorPC, p.MeanColor] = sample_colormodel(p.imagesize);
        end
        if p.impose_spectrum
            p.Magnitude = sample_1f_spectrum(p.imagesize, p.sloperange);
        end
        if p.impose_wmm
            [p.Histb, p.HistbBins, p.meanb, p.stdb] = sample_wmm_model(p.imagesize, p.Nor, p.Nscales);
        end
        % start with random image
        tmp = 256*rand(p.imagesize, p.imagesize, 3);

        % impose image model
        out = sample_image(tmp, p, 10, p.correlate_channels);

        % show images
        image = uint8(out);
    end

    function generate_synthetic_from_real_stats(real_images_type, sample_is, image_files_list, resolution, base_dataset_dir, ...
            impose_spectrum, impose_wmm, impose_color_model)
    %% Experiment 2: generate synthetic images by using a statistical image model by taking parameters from a real image
    % Collect parameters:
        experiment_dataset_dir = strcat(base_dataset_dir, '/synthetic_from_real_stats_', real_images_type);
        p = image_model.get_common_args(resolution, experiment_dataset_dir, ...
            impose_spectrum, impose_wmm, impose_color_model);
        for sample_i = sample_is
            % read image and resize
            subdir = sprintf('%08d', floor(sample_i / 1000) * 1000);
            folder = strcat(p.dataset_dir, '/', subdir);
            image_file = strcat(folder, sprintf('/img_%06d.png', sample_i));
            if ~isfile(image_file)
                try
                    if ~exist(folder, 'dir')
                       mkdir(folder);
                    end

                    real_image_file = image_files_list{sample_i};
                    splitted_image_file = split(real_image_file, ';' );
                    not_rgb_string = splitted_image_file{1};
                    if strcmp(not_rgb_string, 'not_rgb')
                        continue
                    end
                    real_image_file = splitted_image_file{end};

                    image = image_model.generate_single_image_model_synthetic_from_real_stats(real_image_file, p);
                    if ~isfile(image_file)
                       imwrite(image, image_file);
                    end
                catch exception
                   exception
                   nothing = 0;
                end
            end
        end
    end
    function img = generate_single_image_model_synthetic_from_real_stats(image_file, p)
            img = our_imread(image_file);
            img = double(imresize(img, [p.imagesize p.imagesize]));

            [img_decor, p.colorPC, p.MeanColor] = colorPCA(img);

            % get image parameters, avoid computing if we don't use
            if p.impose_color_model
                [p.HistIma, p.HistImaBins] = get_image_histograms(img_decor);
            end
            if p.impose_spectrum
                p.Magnitude = get_spectrum(img_decor);
            end
            if p.impose_wmm
                [p.Histb, p.HistbBins, p.meanb, p.stdb] = get_wmm(img_decor, p.Nor, p.Nscales);
            end

            % start with random image
            tmp = 256*rand(p.imagesize, p.imagesize, 3);

            % impose image model
            out = sample_image(tmp, p, 10, true);

            img = uint8(out);
    end


    function generate_project_real_to_synthetic_stats(real_images_type, sample_is, image_files_list, resolution, base_dataset_dir, ...
            impose_spectrum, impose_wmm, impose_color_model)
        %% Experiment 3: take a real image and replace its statistical properties by replacing it by the generic model from experiment 1
        % Collect parameters:
        experiment_dataset_dir = strcat(base_dataset_dir, '/project_real_to_synthetic_stats_', real_images_type);
        p = image_model.get_common_args(resolution, experiment_dataset_dir, ...
            impose_spectrum, impose_wmm, impose_color_model);
        for sample_i = sample_is
            % read image and resize
            subdir = sprintf('%08d', floor(sample_i / 1000) * 1000);
            folder = strcat(p.dataset_dir, '/', subdir);
            image_file = strcat(folder, sprintf('/img_%06d.png', sample_i));
            if ~isfile(image_file)
                try
                    if ~exist(folder, 'dir')
                       mkdir(folder);
                    end

                    real_image_file = image_files_list{sample_i};
                    splitted_image_file = split(real_image_file, ';' );
                    not_rgb_string = splitted_image_file{1};
                    if strcmp(not_rgb_string, 'not_rgb')
                        continue
                    end
                    real_image_file = splitted_image_file{end};

                    image = image_model.generate_single_project_real_to_synthetic_stats(real_image_file, p);
                    if ~isfile(image_file)
                       imwrite(image, image_file);
                    end
                catch exception
                   % throw(exception)
                   exception
                   nothing = 0;
                end
            end
        end
    end
    function img = generate_single_project_real_to_synthetic_stats(image_file, p)
        % read image and resize
        img = our_imread(image_file);
        img = double(imresize(img, [p.imagesize p.imagesize]));

        % get image parameters
        [img_decor, p.colorPC, p.MeanColor] = colorPCA(img);

        % Get image_model parameters from our generative statistical image
        % model if we are imposing, else preserve current ones

        if p.impose_color_model
            [p.colorPC, p.MeanColor, p.HistIma, p.HistImaBins, img] = sample_colormodel(p.imagesize);
        else
            [p.HistIma, p.HistImaBins] = get_image_histograms(img_decor);
        end
        if p.impose_spectrum
             p.Magnitude = sample_1f_spectrum(p.imagesize, p.sloperange);
        else
             p.Magnitude = get_spectrum(img_decor);
        end
        if p.impose_wmm
            [p.Histb, p.HistbBins, p.meanb, p.stdb] = sample_wmm_model(p.imagesize, p.Nor, p.Nscales);
        else
            [p.Histb, p.HistbBins, p.meanb, p.stdb] = get_wmm(img_decor, p.Nor, p.Nscales);
        end

        % we set all to impose, so that either the real or the synthetic are imposed
        old_impose_color_model = p.impose_color_model;
        old_impose_spectrum = p.impose_spectrum;
        old_impose_wmm = p.impose_wmm;
        p.impose_color_model = true;
        p.impose_spectrum = true;
        p.impose_wmm = true;

        % impose image model (now we start from the real image that we will modify)
        out = sample_image(img_decor, p, 10, true);

        % restore old impose
        p.impose_color_model = old_impose_color_model;
        p.impose_spectrum = old_impose_spectrum;
        p.impose_wmm = old_impose_wmm;

        img = uint8(out);
    end
   end
end



%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%function out = sample_image(in, HistIma, HistImaBins, Magnitude, Histb, HistbBins, meanb, stdb, Nor, Nscales, N)
function out = sample_image(in, param, Niter, reconstruct_colors)
tmp = in;
for iter = 1:Niter
    % The order in which these statistics are imposed, changes the results.

    % impose Fourier transform
    if param.impose_spectrum
        tmp = impose_spectrum(tmp, param.Magnitude);
    end
    % impose wavelets
    if param.impose_wmm
        tmp = impose_wmm(tmp, param.Histb, param.HistbBins, param.meanb, param.stdb, param.Nor, param.Nscales);
    end
    % impose color histogram
    if param.impose_color_model
        tmp = impose_histogram(tmp, param.HistIma, param.HistImaBins);
    end
end

% Reconstruct original color channels:
if reconstruct_colors
    tmp = inv_colorPCA(tmp, param.colorPC, param.MeanColor);
end

% make image occupy full dynamic range.
tmp = tmp-min(tmp(:));
out = 255*tmp/max(tmp(:));
end

function img = our_imread(image_file)
    img = imread(image_file);
    if size(size(img), 2) == 2
        img = repmat(img, 1, 1, 3);
    end
end

%% PCA Functions
function [out, colorPC, MeanColor] = colorPCA(in)
[m,n,c] = size(in);
for i=1:3
    MeanColor(i)=mean(mean(in(:,:,i)));
    in(:,:,i)=in(:,:,i)-MeanColor(i);
end

% PCA of color space to have decorrelated color channels:
x = reshape(in, [m*n 3]);
X=x'*x;
[colorPC,latent] = eigs(X,3);
x=x*colorPC;
out = reshape(x, [m n 3]);
end

function out = inv_colorPCA(in, colorPC, MeanColor)
[m,n,c] = size(in);
in = reshape(in, [m*n c]);
out = in*colorPC';
out = reshape(out,[m n c]);

for i=1:c
    out(:,:,i)=out(:,:,i)+MeanColor(i);
end
end

%% Image histograms:
function [HistIma, HistImaBins] = get_image_histograms(in)
% Collect parameters:
nbins = 256;
for i=1:3
    [H,bins]=histo(reshape(in(:,:,i).',1,[]),nbins);
    HistIma(:,i)=H';
    HistImaBins(:,i)=bins';
end
end

function out = impose_histogram(in, HistIma, HistImaBins)
nbins = length(HistIma);

for i=1:3
    out(:,:,i) = histoMatch(in(:,:,i), HistIma(:,i), HistImaBins(:,i));
end
end


%% Image spectrum
function Magnitude = get_spectrum(in)
for i = 1:3
    Magnitude(:,:,i) = abs(fft2(in(:,:,i)));
end
end

function out = impose_spectrum(in, Magnitude)
% [m,n,c] = size(in);
% slope = 1;
%
% [fx,fy] = meshgrid([1:n]-(n+1)/2, [1:m]-(m+1)/2);
% fr = (eps+abs(fx/n) + abs(fy/m)).^(slope);
% Magnitude = 1./ fftshift(fr);

for i = 1:3
    x = in(:,:,i);

    m = mean(x(:));
    s = std(x(:));

    FFTtexture = fft2(x-m);
    texturePhase = angle(FFTtexture);

    y = real(ifftn(Magnitude(:,:,i) .* exp(sqrt(-1)*texturePhase)));
    y = y-mean(y(:));
    y = y/std(y(:))*s+m;
    out(:,:,i) = y;

end
    %keyboard
end

%% marginal wavelet models
function [Histb, HistbBins, meanb, stdb] = get_wmm(in, Nor, Nscales)
% Collect wavelet marginal histograms:
pyrFilters = 'sp3Filters'; edges='reflect1';
Nbins=251; % number of bins for computing histograms

for i=1:3
    [examplePyr, exampleIndT] = buildSpyr(in(:,:,i), Nscales, pyrFilters, edges);

    % Histogram of the subbands:
    for b=1:Nscales*Nor+2
        x =  pyrBand(examplePyr,exampleIndT,b);
        % x=x(5:end-5,5:end-5); % remove boundary artifacts?
        %[shat(b),rhat(b),y0(b)] = FitLaplacian(x);

        [H, bins]=histo(x, Nbins);
        Histb(:,b,i)=H';
        HistbBins(:,b,i)=bins';
        meanb(b,i) = mean(x(:));
        stdb(b,i) = std(x(:));
    end
end

end

function out = impose_wmm(in, Histb, HistbBins, meanb, stdb, Nor, Nscales)
pyrFilters = 'sp3Filters'; edges='reflect1'; edges='circular';

for i=1:3
    texture=in(:,:,i);    
    [texturePyr, textureIndT] = buildSpyr(texture, Nscales, pyrFilters, edges);
    
    % Coerce subband statistics:
    % the first pyr is the high-pass residual and the last one is the
    % low-pass residual.
    for b=1:Nscales*Nor+1
        x = pyrBand(texturePyr,textureIndT,b);
        x = (x-mean(x(:)))/std(x(:));
        x = x*stdb(b,i) + meanb(b,i);
        
        x = histoMatch(x, Histb(:,b,i), HistbBins(:,b,i));
        x = (x-mean(x(:)))/std(reshape(x(:,:).',1,[]));
        x = x*stdb(b,i) + meanb(b,i);
        
        indices = pyrBandIndices(textureIndT, b);
        texturePyr(indices)=x(:);
    end
    texture = reconSpyr(texturePyr, textureIndT, pyrFilters, edges);
    
    % put texture in the corresponding color channel:
    out(:,:,i)=texture;
end
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODELS

% Model of the Spectrum of natural images
function Magnitude = sample_1f_spectrum(imgsize, slope)

s = slope(1) + rand*(slope(2)-slope(1));
d = randn*abs(slope(2)-slope(1))/15/4;
slopex = s+d;
slopey = s-d;
disp(sprintf('Spectrum model: slopex=%2.2f, slopey=%2.2f', s, d))

[fx,fy] = meshgrid(0:imgsize-1);
fx = fx-imgsize/2;
fy = fy-imgsize/2;

fr = (eps+abs(fx/imgsize).^(slopex) + abs(fy/imgsize).^(slopey));
Magnitude = 1./ fftshift(fr);
Magnitude(1,1)=0; % set to zero the gain in the mean;
Magnitude = repmat(Magnitude, [1 1 3]);
end

% Model of image color distributions:
function [colorPC, MeanColor, HistIma, HistImaBins, ref] = sample_colormodel(imgsize)
%
% This model assumes that the image has N distinct colors. 

% http://perso-laris.univ-angers.fr/~chapeau/papers/chaosf09.pdf
%[img_decor, colorPC, MeanColor] = colorPCA(img);
%[HistIma, HistImaBins] = get_image_histograms(img_decor);
% Create a mondrian image and get its PCA and statistic

Ncolors = 3+floor(20*rand); % number of colors that we will have
colors = rand([Ncolors 3]);
P = 1:Ncolors;
P = 0.001+rand(Ncolors,1);
P = P / sum(P); % P contains the proportion of pixels for each color 
X = mnrnd(1, P, imgsize^2);
X = X * [1:Ncolors]'; % X is the color indices for all the pixels

X = sort(X);
X = reshape(X, [imgsize imgsize]);
ref = 256*ind2rgb(X-1, colors); % ref is the generated color image that we will use as reference to get the color pca and histograms.

m = ceil(rand*10);
ref = convn(ref, ones(m,m)/m/m, 'same');
ref = ref + 10*rand(size(ref));
ref = ref - min(ref(:));
ref = 256*ref/max(ref(:));

[img_decor, colorPC, MeanColor] = colorPCA(ref);
[HistIma, HistImaBins] = get_image_histograms(img_decor);

end


function [Histb, HistbBins, meanb, stdb] = sample_wmm_model(imgsize, Nor, Nscales)
texture = rand([imgsize, imgsize]);
pyrFilters = 'sp3Filters'; edges='reflect1'; edges='circular';
[texturePyr, textureIndT] = buildSpyr(texture, Nscales, pyrFilters, edges);

x = -200:0.5:200;
for i = 1:3
    for b=1:Nscales*Nor+1
        scale = imgsize/textureIndT(b,1);
        shat = 4^scale;
        rhat = 0.4 + 0.4*rand;
        h = exp(-abs(x/shat).^rhat);
        
        m = sum(x.*h)/sum(h);
        s = sqrt(sum((x-m).^2.*h)/sum(h));

        HistbBins(:,b,i) = x;
        Histb(:,b,i) = h;
        meanb(b,i) = m;
        stdb(b,i) = s;        
    end
end

end
