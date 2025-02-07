
for task in deblurring inpainting superresolution; do
    for start_time in 0 ; do
        for mc_batch_size in 4096; do
            python run/inference_inverse.py \
                --guide_method MC \
                --problem $task \
                --mc_batch_size $mc_batch_size \
                --start_time $start_time 
        done
    done
done