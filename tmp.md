!python /kaggle/working/SAPGraph/train_SAP.py --cuda --gpu 0 --data_dir /kaggle/input/sapg-input --cache_dir /kaggle/working/SAPGraph/cache/cordSum --embedding_path /kaggle/input/glove42b300dtxt/glove.42B.300d.txt --model [HSG] --save_root /kaggle/working/ --log_root /kaggle/working/log --lr_descent --grad_clip -m 3


python train.py --cuda --gpu 0 --data_dir <data/dir/of/your/json-format/dataset> --cache_dir <cache/directory/of/graph/features> --embedding_path <glove_path> --model [HSG|HDSG] --save_root <model path> --log_root <log path> --lr_descent --grad_clip -m 3


!python /kaggle/working/SAPGraph/train_SAP.py --cuda --gpu 0 --data_dir /kaggle/input/sapg-input --cache_dir /kaggle/working/SAPGraph/cache/cordSum --embedding_path /kaggle/input/glove42b300dtxt/glove.42B.300d.txt --model HSG --save_root /kaggle/working/ --log_root /kaggle/working/log --lr_descent --grad_clip -m 3


!python /kaggle/working/HeterSumGraph/train.py --cuda --gpu 0 --data_dir /kaggle/input/hsg-dataset/multinews --cache_dir /kaggle/working/HeterSumGraph/cache/multinews --embedding_path /kaggle/input/glove42b300dtxt/glove.42B.300d.txt --model HSG --save_root /kaggle/working/ --log_root /kaggle/working/log --lr_descent --grad_clip -m 3

python /kaggle/working/HeterSumGraph/train.py \
  --data_dir="/kaggle/input/hsg-dataset/multinews" \
  --cache_dir="/kaggle/working/HeterSumGraph/cache/multinews" \
  --embedding_path="/kaggle/input/glove42b300dtxt/glove.42B.300d.txt" \
  --model="HSG" \
  --save_root="save/" \
  --log_root="log/" \
  --n_epochs=2 \
  --lr=1e-4 \
  --batch_size=16 \
  --cuda=True \
  --grad_clip=True \
  --max_grad_norm=1.0 \
  --restore_model="None"
