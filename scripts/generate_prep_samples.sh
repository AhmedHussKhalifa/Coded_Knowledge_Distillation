GPU_ID=0

teacher_list=(  
				# "wrn_40_2" \
				# "resnet56"\
				"resnet32x4" \
				# "ResNet50" \
				# "resnet110" \
				# "vgg13"
				)


for model in ${teacher_list[*]}; do
    echo ${model}
	model_path=./save/models/${model}_vanilla/ckpt_epoch_240.pth
   # Command to run Python script
	CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 generate_ckd.py --path_t ${model_path} --ckd ckd --delta 5
done
