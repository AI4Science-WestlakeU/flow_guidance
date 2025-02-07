
for flow_matching_type in cfm ot_cfm; do
    for env in halfcheetah hopper walker2d; do
        for horizon in 20; do
            for dataset in "medium" "medium-replay" "medium-expert"; do
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

                python run/train.py \
                    --device cuda:0 \
                    --log_folder ./logs \
                    --exp_name "$flow_prefix"H"$horizon"_1e6steps \
                    --env $env-$dataset-v2 \
                    --horizon $horizon \
                    --state_dim $state_dim \
                    --action_dim $action_dim \
                    --n_train_steps 1000001 \
                    --save_freq 50000 \
                    --lr_schdule_T 1000000 \
                    --batch_size 32 \
                    --learning_rate 2e-4 \
                    --ema_decay 0.995 \
                    --flow_matching_type $flow_matching_type
            done
        done
    done
done