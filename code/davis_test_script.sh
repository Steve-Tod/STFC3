# This is an example command for DAVIS evaluation, cropSize -1 is for 480p(JF=56.8 using our model), you need to modify the code for 720p evaluation in our paper.
python test.py --filelist eval/davis_vallist.txt --model-type scratch --resume $1 --save-path /homes/55/yansong/$2 --topk 10 --videoLen 20 --radius 12  --temperature 0.05  --cropSize $4 --gpu-id $3
python eval/convert_davis.py --in_folder /homes/55/yansong/$2/ --out_folder /homes/55/yansong/davis_results/$2_convert/ --dataset /homes/55/yansong/davis-2017/DAVIS/
python /homes/55/yansong/CRWFCN-master/code/davis2017-evaluation/evaluation_method.py --task semi-supervised   --results_path /homes/55/yansong/davis_results/$2_convert/ --set val --davis_path /homes/55/yansong/davis-2017/DAVIS/
