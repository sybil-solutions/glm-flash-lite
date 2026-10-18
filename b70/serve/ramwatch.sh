#!/bin/bash
# ramwatch.sh CONTAINER OUT.csv: host-RAM accounting for one server container, once a second (run on the host; start it
# BEFORE `docker run` so the first row is the baseline). Columns:
#   mem_available_gb       system MemAvailable (/proc/meminfo)
#   cg_current_gb, cg_peak_gb   the container's cgroup memory.current / memory.peak (anon + page cache + shmem it is
#                          charged for; the xe driver's USM host allocations are NOT charged here)
#   pss_anon_gb, pss_shmem_gb, pss_file_gb   summed smaps_rollup PSS of every process in the container
#   held_gb                pss_anon + pss_shmem (what the server holds that the kernel cannot reclaim)
# After the run: uncharged (USM + driver) ~= (baseline MemAvailable - MemAvailable) - (cg_current - reclaimable file).
# Every external call is bounded (timeout 10).
C=$1; OUT=${2:-ramwatch.csv}
echo "t,mem_available_gb,cg_current_gb,cg_peak_gb,pss_anon_gb,pss_shmem_gb,pss_file_gb,held_gb" > "$OUT"
while true; do
  t=$(date +%s)
  ma=$(awk '/MemAvailable/{printf "%.2f", $2/1e6}' /proc/meminfo)
  cg=$(timeout 10 docker exec "$C" sh -c 'cat /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak 2>/dev/null' 2>/dev/null |
       awk 'NR==1{c=$1} NR==2{p=$1} END{if (c) printf "%.2f,%.2f", c/1e9, p/1e9; else printf ","}')
  ps=$(timeout 10 docker exec "$C" sh -c 'cat /proc/[0-9]*/smaps_rollup 2>/dev/null' 2>/dev/null |
       awk '/^Pss_Anon:/{a+=$2} /^Pss_Shmem:/{s+=$2} /^Pss_File:/{f+=$2} END{if (a+s+f) printf "%.2f,%.2f,%.2f,%.2f", a/1e6, s/1e6, f/1e6, (a+s)/1e6; else printf ",,,"}')
  echo "$t,$ma,${cg:-,},${ps:-,,,}" >> "$OUT"
  sleep 1
done
