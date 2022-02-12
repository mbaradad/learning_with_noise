addpath ('matlabPyrTools')

N_samples_256 = 1300000;
N_samples_128 = 105000;

dataset_dir = 'generated_datasets/matlab_statistical_models';

% for multimachine, we randomize list so that each machine processes at
% random, we check file not written before generating and before saving
sample_list_256 = randomize_list(0:N_samples_256 - 1);
sample_list_128 = randomize_list(0:N_samples_128 - 1);

% impose_spectrum, impose_wmm, impose_color_model, jpg

% dataset 1 Only spectrum
image_model.generate_projection_dataset(sample_list_256, 256, dataset_dir, true, false, false, true)
image_model.generate_projection_dataset(sample_list_128, 128, dataset_dir, true, false, false, true)

% dataset 2 Only Wmm
image_model.generate_projection_dataset(sample_list_256, 256, dataset_dir, false, true, false, true)
image_model.generate_projection_dataset(sample_list_128, 128, dataset_dir, false, true, false, true)

% dataset 3 Only spectrum + Color
image_model.generate_projection_dataset(sample_list_256, 256, dataset_dir, true, false, true, true)
image_model.generate_projection_dataset(sample_list_128, 128, dataset_dir, true, false, true, true)

% dataset 4 Color + spectrum + wmm
image_model.generate_projection_dataset(sample_list_256, 256, dataset_dir, true, true, true, true)
image_model.generate_projection_dataset(sample_list_128, 128, dataset_dir, true, true, true, true)


function randomized_list = randomize_list(v)
    d = datetime('now');
    seed = mod(posixtime(d) * 1000000, 1000000);
    rand('seed', seed); randn('seed', seed);
    randomized_list = v(randperm(length(v)));
end