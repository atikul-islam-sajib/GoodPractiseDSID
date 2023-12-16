#!/bin/bash

# Create a new conda environment named alzheimer

conda create -n alzheimer python=3.9 -y

conda activate alzheimer

#!/bin/bash

# Navigate to the GoodPractiseDSID project directory

cd /path/to/GoodPractiseDSID

# Run the classifier script with specified arguments

python ./alzheimer/classifier/classifier.py --dataset /path/to/dataset.zip --model --epochs 300 --lr 0.001 --batch_size 128 --device mps

Upload more soon ....
