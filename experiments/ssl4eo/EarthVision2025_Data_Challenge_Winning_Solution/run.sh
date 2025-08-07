conda init
conda env create --name cvpr_comp --file=env.yaml
conda activate cvpr_comp

############## Open-Sourced Models / Off-the-Shelf GeoRS-CLIP ##############

cd ./GEORSCLIP

git clone https://github.com/mlfoundations/open_clip.git
pip install -e open_clip

python embed_georsclip_step1.py --dataset-path /mnt/disk3/SSL4EO_TEST/data_eval --ckpt-path ../ckpts/RS5M_ViT-H-14.pt

python embed_georsclip_cluster_step2.py -i ./georsclip-embeddings-step1.pt -d /mnt/disk3/SSL4EO_TEST/data_eval -o ../submissions/georsclip_submission_512.csv



############## Custom OpenCLIP Models / Finetuned on SSL4EO  ##############

cd ..

git clone https://github.com/mlfoundations/open_clip.git

# include custom changes here
cp open_clip_custom/convnext_xxlarge.json open_clip/src/open_clip/model_configs/
cp open_clip_custom/ViT-H-14.json open_clip/src/open_clip/model_configs/
cp open_clip_custom/factory.py open_clip/src/open_clip/
cp open_clip_custom/timm_model.py open_clip/src/open_clip/
cp open_clip_custom/transform.py open_clip/src/open_clip/
cp open_clip_custom/transformer.py open_clip/src/open_clip/

pip install -e open_clip


############## CLIP & DINO Embedding Generation ##############

python embed_vitH.py --dataset-path /mnt/disk3/SSL4EO_TEST/data_eval --ckpt-path ./ckpts/epoch_25.pt -o ./submissions/embeddings_clip_VITH_256_E25.csv

python embed_cnxtxxlarge.py --dataset-path /mnt/disk3/SSL4EO_TEST/data_eval --ckpt-path ./ckpts/epoch_22.pt -o ./submissions/embeddings_clip_convnext_xxlarge_256_ftregress_e22.csv

python embed_vitb_dino.py --dataset-path /mnt/disk3/SSL4EO_TEST/data_eval --ckpt-path ./ckpts/checkpoint0030.pth -o ./submissions/embedding_vit_base_finetune_epoch0030.csv



############## SVD compressive ensemble ##############

python final_merge.py 



### final submission csv path: ./submissions/cxxft_vith_dinobfttest_4season512.csv