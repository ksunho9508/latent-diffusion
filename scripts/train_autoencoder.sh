python main.py --base configs/autoencoder/autoencoder_vq_64x64x3.yaml -t --gpus 0,1,2,3 &&
python main.py --base configs/autoencoder/autoencoder_vq_32x32x4.yaml -t --gpus 0,1,2,3&&
python main.py --base configs/autoencoder/autoencoder_vq_16x16x16.yaml -t --gpus 0,1,2,3 &&
python main.py --base configs/autoencoder/autoencoder_vq_8x8x64.yaml -t --gpus 0,1,2,3