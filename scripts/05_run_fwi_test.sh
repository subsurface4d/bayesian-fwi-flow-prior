#!/bin/bash

# >>> run from repo root so `model/...` and `src/` resolve (added during reorg) <<<
cd "$(dirname "$0")/.." || exit 1

# Start timer
SECONDS=0
echo "🚀 Starting all jobs at $(date)"

jobs_in_batch=0
noise_snr=None
src_id=2
rec_id=1
freq=30
i=1

# Construct output directory based on whether noise is used
if [ "$noise_snr" == "None" ]; then
    out_dir="/net/vision/scr2/haipeng/FWI-HMC/Runs-Paper/SYN-survey-${src_id}-geophone-${rec_id}-${freq}Hz-test"
    noise_flag=""
else
    out_dir="/net/vision/scr2/haipeng/FWI-HMC/Runs-Paper/SYN-survey-${src_id}-geophone-${rec_id}-${freq}-noise-${noise_snr}dB-test"
    noise_flag="--noise_snr $noise_snr"
fi

# Run the job
python src/HMC_FWI_VAE.py --output_dir "$out_dir"  --f0 $freq --src_id $src_id \
                        --vp_ml_file model/vp_ml_nz346_nx401_5m.npy \
                        $noise_flag  --device cuda:$((i - 1))


# Print total run time
duration=$SECONDS
echo "✅ All jobs finished at $(date)"
echo "⏱️ Total runtime: $((duration / 3600))h $(( (duration % 3600) / 60))m $((duration % 60))s"


# To kill the running processes, use:
# pkill -f HMC_FWI_VAE.py
