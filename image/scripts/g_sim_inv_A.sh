
for task in deblurring inpainting superresolution; do
    for start_time in 0 ; do
        for schedule in const linear_decay cosine_decay; do
            for scale in 0.1 1.0 10.0; do
                for rt_schedule in linear_decay const linear_ramp_rt; do
                    for rt_scale in 0.1 1.0 10.0; do
                        python run/inference_inverse.py \
                            --guide_method PiGDM+ \
                            --rt_schedule $rt_schedule \
                            --rt_scale $rt_scale \
                            --schedule $schedule \
                            --guide_scale $scale \
                            --problem $task \
                            --start_time $start_time \
                            --time_ratio_eps 1e-2 \
                            --no-schedule_ratio \
                            --no-clamp_x
                    done
                done
            done
        done
    done
done