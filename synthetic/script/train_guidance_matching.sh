
for pair in "circle s_curve" "8gaussian moon" "uniform 8gaussian"; do
    set -- $pair
    x0_dist=$1
    x1_dist=$2
    python guided_flow/train/value_guidance_matching.py --x0_dist $x0_dist --x1_dist $x1_dist
done