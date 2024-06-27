GPU0=2
GPU1=3
GPU2=4
GPU3=5

PYTHONPATH=.

python stylegan/generate_dataset.py --gpu $GPU0 --nimg 325000 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name /mnt/localssd1/jwulff/datasets/stylegan-large-oriented-uniformbias0.2 &
python stylegan/generate_dataset.py --gpu $GPU1 --startimg 325000 --nimg 325000 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name /mnt/localssd1/jwulff/datasets/stylegan-large-oriented-uniformbias0.2 &
python stylegan/generate_dataset.py --gpu $GPU2 --startimg 650000 --nimg 325000 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name /mnt/localssd1/jwulff/datasets/stylegan-large-oriented-uniformbias0.2 &
python stylegan/generate_dataset.py --gpu $GPU3 --startimg 975000 --nimg 325000 --network_type sparse_new --random_configuration 'chin-chout' --same_noise_map --bias_range 0.2 --name /mnt/localssd1/jwulff/datasets/stylegan-large-oriented-uniformbias0.2 &

wait

