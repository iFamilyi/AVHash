# AVHash
This repository is the PyTorch implementation of ACM MM 2024 (CCF A) paper: "AVHash:Joint Audio-Visual Hashing for Video Retrieval".

## Catalogue

- [Getting Started](#getting-started)
- [Data Processing](#data-processing)
- [Train](#train)
- [Test](#test)
  
## Getting Started
1. Clone this repository:
```
git clone https://AVHash.git
cd AVHash
```
2. Create a conda environment and install the dependencies:
```
conda create -n avhash python=3.10
conda activate avhash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data Processing
1. Download Datasets
   Datasets of FCVID and ActivityNet are kindly uploaded by the authors. You can download them from the following links.
   
	| *Dataset*   | *Link*                                                  |
	| ----------- | ------------------------------------------------------- |
	| FCVID       | [Link](https://fvl.fudan.edu.cn/dataset/fcvid/list.htm) |
	| ActivityNet | [Link](http://activity-net.org/)                        |

2. Feauture Extraction

    2.1 **Video Processing**:
    - For each video, uniformly extract 25 frames.
    - Uniformly divide the audio into 25 segments.

    2.2 **Feature Extraction**:

    - For each frame, use the [CLIP](https://github.com/openai/CLIP) model to generate a 768-dimensional vector.
    - For each audio segment, use the [AST](https://github.com/YuanGongND/ast) model to generate a 768-dimensional vector.

    2.3 **Data Storage**:
    - Each video has `(25, 768)` image features and `(25, 768)` audio features.
    
    - Store the features of each video in one HDF5 file.
    
    - Store all audio features in another HDF5 file.
  
      The structure of the HDF5 file storing video features is as follows:
      ```
      ActivityNet_image.h5
      ├── v_---9CpRcKoU
      │ └── vectors
      │ └── Type: float32
      ├── v_--0edUL8zmA
      │ └── vectors
      │ ├── Shape: (25, 768)
      │ └── Type: float32
      ├── v_--1DO2V4K74
      │ └── vectors
      │ ├── Shape: (25, 768)
      │ └── .....
      └── .....
      ```
      
      The structure of the HDF5 file storing audio features is as follows:
       ```
      ActivityNet_audio.h5
      ├── v_---9CpRcKoU
      │ └── vectors
      │ └── Type: float32
      ├── v_--0edUL8zmA
      │ └── vectors
      │ ├── Shape: (25, 768)
      │ └── Type: float32
      ├── v_--1DO2V4K74
      │ └── vectors
      │ ├── Shape: (25, 768)
      │ └── .....
      └── .....
      ```

	
      
      In above structure:
      - Each video's ID serves as the top-level group name.
      - Each group contains a dataset named vectors.
      - The vectors dataset contains the image or audio features of the video, with a shape of (25, 768) and a type of float32.
​
3. Dataset Splitting
   Split the dataset into training, validation, and test sets evenly based on categories. The file IDs for different data splits are stored in `train.txt`, `test.txt`, and `val.txt` files respectively.
   
   **ActivityNet** contents are as follows, with each line including: videoname, video frame count, category.
   ```
   v_JDg--pjY5gg,3153,10
   v_DFAodsf1dWk,6943,10,10,10
   v_J__1J4MmH4w,3370,10,10,10,10
   ...
   ```

	 **FCVID** contents are as follows, with each line including: videoname, category.
   ```
   --0K_j-zexM,76
   --1DKnUmLNQ,163
   --45hTBwKRI,117
   ...
   ```
   
4. Configure the **Anet.json** and **fcvid.json** file in ./Json/
   ```
   {
   "dataset":  dataset ("actnet" or "fcvid")
   "data_path": path to the image frames floder,
   "num_class":  dataset classes, 200(actnet) or 239(fcivd)
   "train_list": "path to the train set file",
   "val_list": "path to the validation set file",
   "test_list": "path to the test set file",
   "retrieval_list": "path to the databese set (train set) file"
   }
   ```
   


  ## Train

  To train AVHash on FCVID:

  ```bash
  python run.py --dataset "path to fcvid.json" --hashcode_size 64 --lr le-4 --max_iter 100 --device cuda:0 --batch_size 128 --result_log_dir " " --result_weight_dir " "
  ```

  To train AVHash on ActivityNet:

  ```bash
  python run.py --dataset "path to Anet.json" --hashcode_size 64 --lr le-4 --max_iter 100 --device cuda:0 --batch_size 128 --result_log_dir " " --result_weight_dir " "
  ```

  Options:

  - `--dataset`: path to Anet.json or fcvid.json
  - `--hashcode_size`: code length of hash.
  - `--lr`: learning rate.
  - `--max_iter`: training epochs.
  - `--device`: choose the gpu to use.
  - `--batch_size`: the number of videos in a batch.
  - `--result_log_dir`: path to save log
  - `--result_weight_dir`: path to save weight


  ## Test
  To test AVHash on FCVID:
  ```bash
  python test.py --dataset "path to fcvid.json" --hashcode_size 64 --device cuda:0 --batch_size 128 --weight_path " "
  ```
  To test AVHash on ActivityNet: 
  ```bash
  python test.py --dataset "path to Anet.json" --hashcode_size 64 --device cuda:0 --batch_size 128 --weight_path " "
  ```

  Options:

  - `--dataset`: path to Anet.json or fcvid.json
  - `--hashcode_size`: code length of hash.
  - `--device`: choose the gpu to use.
  - `--weight_path`: the path to the weights to be loaded.
  - `--batch_size`: the number of videos in a batch.
  - `--weight_path`: the path to the weights to be loaded.

 
