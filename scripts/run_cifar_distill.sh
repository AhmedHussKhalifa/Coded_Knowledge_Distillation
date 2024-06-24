# sample scripts for running the distillation code
# use resnet32x4 and resnet8x4 as an example

model=resnet56


model_path=./save/models/${model}_vanilla/ckpt_epoch_240.pth

student_list=(  
				"resnet20" \
				# "resnet8x4"\
				# "MobileNetV2" \
				)

method_list=(  
				# "kd" \
				"hint"\
				# "attention" \
				"similarity" \
				"correlation" \
				# "vid" \
				"rkd" \
				# "pkt" \
				"abound" \
				"factor" \
				"fsp" \
				# "nst" \
				)

parameters_list=(
	# "-r 0.1 -a 0.9 -b 0" \
	"-a 0 -b 100" \
	# "-a 0 -b 1000 " \
	"-a 0 -b 3000" \
	"-a 0 -b 0.02" \
	# "-a 0 -b 1" \
	"-a 0 -b 1" \
	# "-a 0 -b 30000" \
	"-a 0 -b 1" \
	"-a 0 -b 200" \
	"-a 0 -b 50" \
	# "-a 0 -b 50" \
)

# Get the length of the lists
len=${#method_list[@]}

for ((i=0; i<$len; i++)); do
	for student in ${student_list[*]}; do
	echo ${model}
	# Command to run Python script
		CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_student.py --path_t ${model_path} \
										--distill ${method_list[$i]} --model_s ${student}  \
										--num_workers 8 ${parameters_list[$i]} \
										--trial 1
	done
done
