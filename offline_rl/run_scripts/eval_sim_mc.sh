conda activate gflower

for flow_matching_type in cfm ot_cfm; do
    for env in halfcheetah hopper walker2d; do
        for dataset in "medium-expert" "medium-replay" "medium"; do
            for B in 16 64; do
                for scale in 0.1 1; do
                    for ep in 1e-2; do
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
                        python run/eval.py \
                            --device cuda:0 \
                            --seed 0 \
                            --random_repeat 5 \
                            --exp_name "$flow_prefix"H20_1e6steps_mc_10steps_inf_"$B"_scale_"$scale"_ep_"$ep"_ss_"$ss_b" \
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
                            --guidance_method sim_mc \
                            --sim_mc_n 100 \
                            --sim_mc_J_scale 1.0 \
                            --sim_mc_std 0.5 \
                            --sim_mc_eps 1e-2 \
                            --sim_mc_scale $scale \
                            --sim_mc_self_normalize
                    done
                done
            done
        done
    done
done
