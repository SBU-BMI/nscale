grep -rl "sessionName,io" * | xargs grep -l "compute,0," | xargs sed -ibak -s 's/compute,0,[0-9]\+,[0-9]\+,1,//g'
