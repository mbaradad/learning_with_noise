GPU0=2
GPU1=3
GPU2=4
GPU3=5

PYTHONPATH=.

python stylegan/generate_dataset.py --gpu $GPU0 --res 512 --nimg 325000 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name /vision-nfs/torralba/env/mbaradad/movies_sfm/releases/noiselearning/raw_data/stylegan-oriented-512 &
python stylegan/generate_dataset.py --gpu $GPU1 --res 512 --startimg 325000 --nimg 325000 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name /vision-nfs/torralba/env/mbaradad/movies_sfm/releases/noiselearning/raw_data/stylegan-oriented-512 &
python stylegan/generate_dataset.py --gpu $GPU2 --res 512 --startimg 650000 --nimg 325000 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name /vision-nfs/torralba/env/mbaradad/movies_sfm/releases/noiselearning/raw_data/stylegan-oriented-512 &
python stylegan/generate_dataset.py --gpu $GPU3 --res 512  --startimg 975000 --nimg 325000 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name /vision-nfs/torralba/env/mbaradad/movies_sfm/releases/noiselearning/raw_data/stylegan-oriented-512 &

wait

# python generate_dataset.py --nimg 1300 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name generated_datasets/stylegan-oriented