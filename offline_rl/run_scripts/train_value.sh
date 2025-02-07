conda activate gflower

for env in halfcheetah hopper walker2d; do
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

        python run/train_value.py \
            --device cuda:0 \
            --exp_name H20_inf \
            --env $env-$dataset-v2 \
            --inf_horizon \
            --horizon 20 \
            --state_dim $state_dim \
            --action_dim $action_dim \
            --n_train_steps 10001 \
            --save_freq 5000 \
            --batch_size 64 \
            --learning_rate 2e-4 
    done
done