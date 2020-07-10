# Using download links from https://github.com/facebookresearch/odin
cd data

# Download TinyImageNet (cropped)
wget https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz
tar -xvzf Imagenet.tar.gz
rm Imagenet.tar.gz

# Download TinyImageNet (resized)
wget https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz
tar -xvzf Imagenet_resize.tar.gz
rm Imagenet_resize.tar.gz
