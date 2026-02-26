for k in 1 2 3 4 5 10 15 20 25 30 35 40 45 50 60 70 80 90 100; do
  python reg_path_pglasso.py --max_iter_reweights "$k"
done