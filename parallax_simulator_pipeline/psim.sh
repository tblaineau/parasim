#!/bin/sh

#$-N prod_parallax_distance_u02_minmax_run2
#$-P P_eros
#$-t 1-100:1
#$-q long
#$-o $HOME/work/parallax_jobs/run2/logs_minmax
#$-e $HOME/work/parallax_jobs/run2/logs_minmax

SGE_TASK_ID=1

#"$HOME/work/merger/merger/parallax_estimator/se_script.py"
python3 "se_script.py" --name "$1_$SGE_TASK_ID.pkl" --parameter_file "$2" \
--nb_line_pmsfile "$3" --nb_files "$4" --nb_jobs 100  --current_job "$SGE_TASK_ID"

for var in *.pkl
do
        echo "$var"
        cp "$var" "$HOME/work/parallax_jobs/run2/minmax"
done