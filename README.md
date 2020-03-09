# Disrupting Deepfakes: Adversarial Attacks on Conditional Image Translation Networks

## StarGAN Testing
```
python main.py --mode test --dataset CelebA --image_size 256 --c_dim 5 --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young --model_save_dir='stargan_celeba_256/models' --result_dir='stargan_celeba_256/results_test' --test_iters 200000 --batch_size 1
```

## StarGAN Training
```
python main.py --mode train --dataset CelebA --image_size 256 --c_dim 5 --sample_dir stargan_both/samples --log_dir stargan_both/logs --model_save_dir stargan_both/models --result_dir stargan_both/results --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
```

## GANimation
```
python main.py --mode animation
```

## pix2pixHD
```
python test.py --name label2city_1024p --netG local --ngf 32 --resize_or_crop none
```

## CycleGAN
```
python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout
```