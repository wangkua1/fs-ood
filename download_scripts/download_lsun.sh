# Using download links from https://github.com/facebookresearch/odin
cd data

# Download LSUN (cropped)
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
tar -xvzf LSUN.tar.gz
rm LSUN.tar.gz

# Download LSUN (resized)
wget https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz
tar -xvzf LSUN_resize.tar.gz
rm LSUN_resize.tar.gz
