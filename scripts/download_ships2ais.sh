wget https://huggingface.co/datasets/isaaccorley/ships-s2-ais/resolve/main/ship-s2-ais-dataset.zip
mkdir ship-s2-ais
unzip ship-s2-ais-dataset.zip -d ship-s2-ais
rm ship-s2-ais-dataset.zip
tar -xvf ship-s2-ais/Sentinel-2-database-for-ship-detection.tar.gz --directory ship-s2-ais
rm ship-s2-ais/Sentinel-2-database-for-ship-detection.tar.gz