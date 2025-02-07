conda activate gflower

device=cuda:0

for flow_matching_type in cfm ot_cfm; do
    for env in halfcheetah hopper walker2d; do
        for scale in 0.5; do
            for dataset in "medium-expert" "medium-replay" "medium"; do
                
                if [ $env == "halfcheetah" ]; then
                    state_dim=17
                    action_dim=6
                elif [ $env == "hopper" ]; then
                    state_dim=11
                    action_dim=3
                elif [ $env == "walker2d" ]; then
                    state_dim=17
                    action_dim=6
                fi

                if [ $flow_matching_type == "cfm" ]; then
                    flow_prefix=""
                elif [ $flow_matching_type == "ot_cfm" ]; then
                    flow_prefix="ot_"
                fi

                # train model to learn z
                python run/train_guide.py \
                    --device $device \
                    --log_folder ./logs \
                    --exp_name "$flow_prefix"H20_scale_"$scale"_v3 \
                    --env $env-$dataset-v2 \
                    --horizon 20 \
                    --state_dim $state_dim \
                    --action_dim $action_dim \
                    \
                    --train_z \
                    --scale $scale \
                    --normed \
                    --guidance_matching_type "" \
                    \
                    --z_transformer_config.depth 4 \
                    --z_transformer_config.num_heads 4 \
                    --z_transformer_config.hidden_size 64 \
                    \
                    --flow_exp_name "$flow_prefix"H20_1e6steps \
                    --flow_cp 19 \
                    --flow_matching_type $flow_matching_type
                
                # then train model to learn g
                # v2: use larger model_g with smaller model_z, larger batch size (1024), and use lr scheduler
                for guidance_matching_type in "direct" "use_learned_v" "rw_use_learned_z" "rw"; do
                    python run/train_guide.py \
                        --device $device \
                        --log_folder ./logs \
                        --exp_name "$flow_prefix"H20_scale_"$scale"_g_${guidance_matching_type}_v5 \
                        --env $env-$dataset-v2 \
                        --horizon 20 \
                        --state_dim $state_dim \
                        --action_dim $action_dim \
                        \
                        --transformer_config.depth 2 \
                        --transformer_config.num_heads 2 \
                        --transformer_config.hidden_size 64 \
                        \
                        --no-train_z \
                        --scale $scale \
                        --normed \
                        --batch_size 1024 \
                        --guidance_matching_type $guidance_matching_type \
                        \
                        --z_exp_name "$flow_prefix"H20_scale_"$scale"_v3 \
                        --z_cp 2 \
                        --z_transformer_config.depth 4 \
                        --z_transformer_config.num_heads 4 \
                        --z_transformer_config.hidden_size 64 \
                        \
                        --flow_exp_name "$flow_prefix"H20_1e6steps \
                        --flow_cp 19 \
                        --flow_matching_type $flow_matching_type \
                        --n_train_steps 2001 \
                        --save_freq 1000 \
                        --batch_size 1024 \
                        --learning_rate 1e-4

                    for inference_scale in 0.01 0.1 1.0 10.0; do
                        python run/eval.py \
                            --device $device \
                            --seed 0 \
                            --random_repeat 5 \
                            --exp_name "$flow_prefix"H20_1e6steps_guidance_matching_10steps_inf_scale_"$scale"_guidance_matching_type_"$guidance_matching_type"_inference_scale_"$inference_scale"_v5 \
                            --env $env-$dataset-v2 \
                            --state_dim $state_dim \
                            --action_dim $action_dim \
                            --horizon 20 \
                            --flow_exp_name "$flow_prefix"H20_1e6steps \
                            --flow_cp 19 \
                            --flow_matching_type $flow_matching_type \
                            --value_exp_name H20_inf \
                            --value_cp 2 \
                            --ode_t_steps 10 \
                            --guidance_method guidance_matching \
                            --guide_matching_type $guidance_matching_type \
                            --guide_scale $scale \
                            --guide_model_exp_name "$flow_prefix"H20_scale_"$scale"_g_"$guidance_matching_type"_v5 \
                            --guide_model_cp 2 \
                            --guide_inference_scale $inference_scale \
                            \
                            --guide_model_transformer_config.depth 2 \
                            --guide_model_transformer_config.num_heads 2 \
                            --guide_model_transformer_config.hidden_size 64
                    done
                done
            done
        done
    done
done