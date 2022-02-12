GPU0=2
GPU1=3
GPU2=4
GPU3=5

cd ..

CUDA_VISIBLE_DEVICES=${GPU0} python generate_dataset.py --nimg 325000 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name /mnt/localssd1/jwulff/datasets/stylegan-large-oriented-uniformbias0.2 &
CUDA_VISIBLE_DEVICES=${GPU1} python generate_dataset.py --startimg 325000 --nimg 325000 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name /mnt/localssd1/jwulff/datasets/stylegan-large-oriented-uniformbias0.2 &
CUDA_VISIBLE_DEVICES=${GPU2} python generate_dataset.py --startimg 650000 --nimg 325000 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name /mnt/localssd1/jwulff/datasets/stylegan-large-oriented-uniformbias0.2 &
CUDA_VISIBLE_DEVICES=${GPU3} python generate_dataset.py --startimg 975000 --nimg 325000 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name /mnt/localssd1/jwulff/datasets/stylegan-large-oriented-uniformbias0.2 &

wait

# python generate_dataset.py --nimg 1300 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name generated_datasets/stylegan-oriented