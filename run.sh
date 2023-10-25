GPU_ID=0

# "wrn_40_2"
# "resnet56"
# "resnet32x4" 

student_list=(  
				"wrn_16_2" \
				# "wrn_40_1"\
				# "ShuffleV1" \
				)

model=wrn_40_2
model_path=./save/models/${model}_vanilla/ckpt_epoch_240.pth

method_list=(  
                # "kd -r 0.1 -a 0.9 -b 0" \
                # "hint -a 0 -b 100"\
                # "attention -a 0 -b 1000" \
                # "similarity -a 0 -b 3000" \
                # "correlation -a 0 -b 0.02" \
                # "vid -a 0 -b 1" \
                # "rkd -a 0 -b 1" \
                "pkt -a 0 -b 30000" \
                # "abound -a 0 -b 1" \
                # "factor -a 0 -b 200" \
                # "fsp -a 0 -b 50" \
                # "nst -a 0 -b 50" \
                )

# Get the length of the lists
len=${#method_list[@]}
for ((i=0; i<$len; i++)); do
    for student in ${student_list[*]}; do
        echo ${model} ${method_list[$i]}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_student_ckd.py --path_t ${model_path} --ckd ckd \
                                        --model_s ${student} --num_workers 16  \
                                        --distill ${method_list[$i]} \
                                        --trial 6


        # CUDA_VISIBLE_DEVICES=${GPU_ID} python3.8 train_student.py --path_t ${model_path} \
        # 								--model_s ${student} --num_workers 16  \
        #                                 --distill ${method_list[$i]} \
        #                                 --trial 1
    done
done