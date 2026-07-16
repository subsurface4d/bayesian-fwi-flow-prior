#!/bin/bash

# >>> run from repo root so `model/...` and `src/` resolve (added during reorg) <<<
cd "$(dirname "$0")/.." || exit 1

# Start timer
SECONDS=0
echo "🚀 Starting all jobs at $(date)"

max_gpu=4
jobs_in_batch=0
idx=0

for noise_snr in None 25 30; do
    for src_id in {1..4}; do
        for rec_id in 1; do
            for freq in 30; do
                    for temp in 0.025; do
                    # Construct output directory based on whether noise is used
                    if [ "$noise_snr" == "None" ]; then
                        out_dir="/net/vision/scr2/haipeng/FWI-HMC-Revision/Runs-Paper/SYN-REG-Smooth-survey-${src_id}-geophone-${rec_id}-${freq}Hz"
                        noise_flag=""
                    else
                        out_dir="/net/vision/scr2/haipeng/FWI-HMC-Revision/Runs-Paper/SYN-REG-Smooth-survey-${src_id}-geophone-${rec_id}-${freq}Hz-noise-${noise_snr}dB"
                        noise_flag="--noise_snr $noise_snr"
                    fi

                    gpu_id=$((idx % max_gpu))

                    # Run the job
                    python src/FWI_REG.py --output_dir "$out_dir"  \
                                    --f0 $freq --src_id $src_id  --rec_id $rec_id \
                                    --vp_ml_file model/vp_ml_nz346_nx401_5m.npy \
                                    $noise_flag                  \
                                    --device "cuda:${gpu_id}"    \
                                    --temp  $temp  &

                    # Wait if max GPU usage is reached
                    idx=$((idx + 1))
                    jobs_in_batch=$((jobs_in_batch + 1))
                    if [ "$jobs_in_batch" -ge "$max_gpu" ]; then
                        wait
                        jobs_in_batch=0
                    fi
                done
            done
        done
    done
done

echo "All jobs started. Use 'wait' to wait for all background jobs to finish."
wait

# Print total run time
duration=$SECONDS
echo "✅ All jobs finished at $(date)"
echo "⏱️ Total runtime: $((duration / 3600))h $(( (duration % 3600) / 60))m $((duration % 60))s"


# To kill the running processes, use:
# pkill -f HMC_FWI_VAE.py
