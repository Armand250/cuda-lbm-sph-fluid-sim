#!/bin/bash
mkdir -p results

SOLVERS=("cpu-naive" "gpu-naive" "gpu-shared")

for config_file in scenes/*.json; do
    [ -e "$config_file" ] || continue
    
    scene_name=$(basename "$config_file" .json)
    if [[ "$scene_name" == _* ]]; then
        continue
    fi
    echo "=========================================================="
    echo " Starting benchmarks for scene: $scene_name"
    echo "=========================================================="
    
    for solver in "${SOLVERS[@]}"; do
        echo "-> Running solver: $solver"
        
        output_dat="results/${scene_name}_${solver}.dat"
        
        if [[ "$solver" == *"gpu"* ]]; then
            profile_out="results/${scene_name}_${solver}_profile"
               ./bin/fluid_sim --config "$config_file" --solver "$solver" \
                --output "$output_dat" --metrics

               ncu --profile-from-start off \
                    --section SpeedOfLight \
                    --section SpeedOfLight_RooflineChart \
                    --section MemoryWorkloadAnalysis \
                    --section ComputeWorkloadAnalysis \
                    --section SourceCounters \
                    --section Occupancy \
                    --force-overwrite \
                    -o "$profile_out" \
                    ./bin/fluid_sim --config "$config_file" --solver "$solver" \
                    --output "$output_dat" --profiling
                
        else
            ./bin/fluid_sim --config "$config_file" --solver "$solver" \
                --output "$output_dat" --metrics
        fi
        
        echo "   Finished $solver for $scene_name."
        echo "----------------------------------------------------------"
    done
    
done

echo "=========================================================="
echo " All benchmarks completed!"
echo "=========================================================="